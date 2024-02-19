import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pickle
import sys
import tqdm

from masp import shoebox_room_sim as srs
from scipy.io import wavfile
from scipy.signal import fftconvolve


# Extra variables
RIRS = {}
SOUND_V = 343
# Values from https://www.acoustic-supplies.com/absorption-coefficient-chart/
MATERIALS = {
    'carpet': [0.08, 0.08, 0.3, 0.6, 0.75, 0.8],  # Frequency bands; 125, 250, 500, 1k, 2k, 4k
    'concrete': [0.36, 0.44, 0.31, 0.29, 0.39, 0.25],
    'drapery': [0.14, 0.35, 0.53, 0.75, 0.7, 0.6],
    'fiberglass': [0.06, 0.2, 0.65, 0.9, 0.95, 0.98],
    'glass': [0.04, 0.04, 0.03, 0.03, 0.02, 0.02],
    'plaster': [0.2, 0.15, 0.1, 0.08, 0.04, 0.02],
    }


def check_and_create_dir(path: str) -> None:
    """ Create directory if it doesn't exist

    :param path: path string (or object?)
    :return:
    """
    if not pathlib.Path(path).exists():
        pathlib.Path(path).mkdir(parents=True)


def create_grid(points: np.array, wall_gap: float, room_dim: np.array) -> np.array:
    """ Create a grid of (x, y) coordinates with specified amount of points (x_points * y_points)

    :param points: grid points in x and y directions, total points will be x*y
    :param wall_gap: minimum distance between walls and grid points
    :param room_dim: room size x*y*z, where z is height
    :return: numpy array with x,y coordinate pairs

    """
    x = np.linspace(wall_gap, room_dim[0] - wall_gap, points[0])
    y = np.linspace(wall_gap, room_dim[1] - wall_gap, points[1])
    xx, yy = np.meshgrid(x, y)

    return np.vstack([xx.ravel(), yy.ravel()]).T


def generate_rir_audio_sh(points: np.array, save_path: str, audio_paths: np.array, heights: np.array = None,
                          room: np.array = None, rt60: float = None, order: int = None, rm_delay: bool = None,
                          args: argparse.Namespace = None) -> None:
    """ Apply spherical harmonics RIR for specified audio at specified points; 
    for each point in the grid RIR applied audio at every other point is generated.
    RIR parameters can be given in args or separately
    Coordinate system (Z direction is 'up' from the screen, i.e. the height):
    ^
    |
    X
    |
    O ⎯ Y ⎯ >

    :param points: coordinate grid (x,y) in the room where RIR is calculated
    :param save_path: folder where to save the created audio files
    :param audio_paths: array with paths to audio dataset wav files
    :param heights: array with source height and listener height, currently heights stay constant for all points
    :param room: room size x*y*z, where z is height
    :param rt60: reverberation time
    :param order: ambisonics order used
    :param rm_delay: true if sound travel time delay should be removed from generated audio
    :param args: startup arguments which can have the required RIR parameters instead of them being given separately
    :return: None

    """
    global RIRS, SOUND_V, MATERIALS
    if args is not None:  # could have checks that other parameters are given if not using args?
        heights = np.array(args.heights)
        room = np.array(args.room)
        rt60 = args.rt60
        order = args.order
        rm_delay = args.rm_delay

    components = (order + 1) ** 2
    # nBands = 1
    # band_centerfreqs = np.empty(nBands)
    # band_centerfreqs[0] = 1000
    band_centerfreqs = np.array([125, 250, 500, 1000, 2000, 4000])
    # abs_wall = srs.find_abs_coeffs_from_rt(room, rt60)[0]  # basic absorption
    abs_wall = np.array([MATERIALS['fiberglass'], MATERIALS['fiberglass'], MATERIALS['fiberglass'],
                         MATERIALS['fiberglass'], MATERIALS['carpet'], MATERIALS['carpet']])
    # rt60 = np.array([rt60])
    rt60, _, _ = srs.room_stats(room, abs_wall, False)
    maxlim = 0.8  # 0.8, 1s
    # limits = np.empty(nBands)
    # limits.fill(np.minimum(rt60[0], maxlim))
    limits = rt60 * 2  # todo: compare with maxlim, although not really needed in practice
    # limits = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    audio_index = 0
    data_index = 0
    minmax = [np.inf, -np.inf]
    progress = tqdm.tqdm(enumerate(points))  # todo: pathlib seems to not work well with eta, possible fixes?
    # First loop to generate RIRs and reverberant audio, then normalize before saving the actual wav files
    for i, src_pos in progress:
        progress.set_description(f'Calculating RIRs for each other point at grid point: {i + 1}/{len(points)}')
        for j, recv_pos in enumerate(points):
            if j == i:
                continue  # skip if source and listener are at the same point

            # Load data
            fs, audio_anechoic = wavfile.read(audio_paths[audio_index])
            # audio_anechoic = np.append(audio_anechoic, np.zeros([400 - len(audio_anechoic) % 400]))  # pad to get equal lengths with coordinate files
            source = np.array([[src_pos[0], src_pos[1], heights[0]]])
            receiver = np.array([[recv_pos[0], recv_pos[1], heights[1]]])
            if rm_delay:
                delay_samples = int(((src_pos[0] - recv_pos[0]) ** 2 + (src_pos[1] - recv_pos[1]) ** 2 + (heights[0] - heights[1]) ** 2) ** (1 / 2) / SOUND_V * fs)

            # Generate/load SH RIRs
            if f'{i}-{j}' in RIRS:
                sh_rirs = RIRS[f'{i}-{j}']
            else:
                abs_echograms = srs.compute_echograms_sh(room, source, receiver, abs_wall, limits, order)
                sh_rirs = srs.render_rirs_sh(abs_echograms, band_centerfreqs, fs).squeeze()
                if order == 0:
                    sh_rirs = sh_rirs.reshape(len(sh_rirs), 1)
                sh_rirs = sh_rirs[500:-1, :]  # Remove delay from filtering, not needed with basic absorption
                sh_rirs = sh_rirs * np.sqrt(4*np.pi) * get_sn3d_norm_coefficients(order)
                if rm_delay:
                    sh_rirs = sh_rirs[delay_samples:-1, :]  # removes last sample as well but it doesn't really matter
                RIRS.update({f'{i}-{j}': sh_rirs})

            # Apply RIRs, check min/max for normalization and store metadata, currently mono is not normalized as it's only used for listening
            subject = data_index + 1
            # audio_length = len(audio_anechoic)  # this can be used to limit reverberant audio length to make sure it matches with mono length and coordinate data
            reverberant_signal = []  # np.zeros((audio_length, components))
            pathlib.Path(f'{save_path}/subject{subject}').mkdir(parents=True)
            wavfile.write(f'{save_path}/subject{subject}/mono.wav', fs, audio_anechoic.astype(np.int16))
            # t = np.arange(len(sh_rirs[:, 0])) / fs
            for k in range(components):
                # plt.plot(t, sh_rirs[:, k])
                reverberant_signal.append(fftconvolve(audio_anechoic, sh_rirs[:, k].squeeze()))
            # plt.show()
            reverberant_signal = np.array(reverberant_signal).T
            if np.min(reverberant_signal) < minmax[0]: minmax[0] = np.min(reverberant_signal)
            if np.max(reverberant_signal) > minmax[1]: minmax[1] = np.max(reverberant_signal)
            # Temporarily store non-normalized reverberant audio, will be very loud
            wavfile.write(f'{save_path}/subject{subject}/ambisonic.wav', fs, reverberant_signal)
            save_coordinates(source=np.array([src_pos[0], src_pos[1], heights[0]]), listener=np.array([recv_pos[0], recv_pos[1], heights[1]]),
                             fs=fs, audio_length=400, path=f'{save_path}/subject{data_index + 1}')  # using 400 to only write the coordinates once

            audio_index += 1
            data_index += 1
            if audio_index == len(audio_paths):  # go through all audio snippets using different one for each data point and start over if needed
                audio_index = 0

    # Save normalized reverberant audio
    max_data = max(abs(minmax[0]), abs(minmax[1]))
    max_path = f'naf/metadata/ambisonics_{args.order}_{args.grid[0]}x{args.grid[1]}/normalization'
    check_and_create_dir(max_path)
    with open(f'{max_path}/{args.room_name}_max_val.txt', 'w') as f:  # todo: paths
        f.write(str(max_data))
    progress = tqdm.tqdm(pathlib.Path(save_path).glob('*'))
    for i, path in enumerate(progress):
        path_str = str(path)
        progress.set_description(f'Normalizing subject {i + 1}')
        fs, reverberant_signal = wavfile.read(f'{path_str}/ambisonic.wav')
        if order == 0:
            reverberant_signal = reverberant_signal.reshape(len(reverberant_signal), 1)
        for k in range(components):
            reverberant_signal[:, k] = normalize(reverberant_signal[:, k], max_data)
        wavfile.write(f'{path_str}/ambisonic.wav', fs, reverberant_signal.astype(np.float32))  # float32 when between -1.0 and 1.0


def get_audio_paths(path: str) -> np.array:
    """ Use TIMIT dataset's train/test_data.csv to get paths to all files used for creating the training data

    :param path: path to TIMIT data
    :return: array with full paths to included wav files
    """
    paths = []
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['is_converted_audio'] == 'TRUE':
                path_start = '/'.join(path.split('/')[:-1])  # get path to timit's data folder from csv file path
                paths.append(f'{path_start}/data/{row["path_from_data_dir"]}')
    return np.array(paths)


def get_sn3d_norm_coefficients(order: int) -> list:
    """ Get list with coefficients for converting N3D norm into SN3D norm

    :param order: ambisonics order
    """
    sn3d = [1]
    i = 1
    while i <= order:
        root = i * 2 + 1  # components "specific" to a certain order will have the same multiplier
        for j in range(root):
            sn3d.append(1. / np.sqrt(root))
        i += 1
    return sn3d


def normalize(data: np.array, max_data: float) -> np.array:
    """ Simple normalization by dividing with the maximum value

    :param max_data: maximum of the whole dataset (not just 'data')
    :param data: array (1d) to normalize
    :return: normalized numpy array
    """
    data_norm = []
    for value in data:
        data_norm.append(value / max_data)
    return np.array(data_norm)


def parse_input_args():
    parser = argparse.ArgumentParser(description='Create reverberant audio dataset encoded into ambisonics with specified order')
    parser.add_argument('-d', '--dataset_path', default='data/timit', type=str, help='path to TIMIT dataset from current parent folder')  # todo: make paths "normal" (?)
    parser.add_argument('-s', '--save_path', default='data/generated', type=str, help='path (from current parent folder) where to save the generated dataset, will be saved in a folder named based on the ambisonics order')
    parser.add_argument('-n', '--naf_path', default='msc/naf/metadata', type=str, help='path (from current parent folder) where to save additional naf metadata')
    parser.add_argument('-r', '--room', nargs=3, default=[10.0, 6.0, 2.5], type=float, help='room size as (x y z)', metavar=('room_x', 'room_y', 'room_z'))  # 20 cm min distance between points
    parser.add_argument('--room_name', default='test_1', type=str, help='name of the room for NAF, use separate name for different sizes')
    parser.add_argument('-g', '--grid', nargs=2, default=[2, 2], type=int, help='grid points in each axis (x y)', metavar=('x_n', 'y_n'))  # todo: give delta instead of points?
    parser.add_argument('-w', '--wall_gap', default=1.0, type=float, help='minimum gap between walls and grid points')
    parser.add_argument('--heights', nargs=2, default=[1.5, 1.5], type=float, help='heights for the source and the listener', metavar=('source_height', 'listener_height'))
    parser.add_argument('--rt60', default=0.2, type=float, help='reverberation time of the room')
    parser.add_argument('-o', '--order', default=1, type=int, help='ambisonics order')
    parser.add_argument('--rm_delay', action='store_true', help='remove travel time delay from generated audio files')
    parser.add_argument('--skip_rir_write', action='store_true', help='by default RIRs are saved according to their parameters under save_path')
    parser.add_argument('--skip_coord_write', action='store_true', help='by default indexes for each coordinate are stored as metadata under save_path')
    parser.add_argument('--skip_split_write', action='store_true', help='allows to skip train/test split info writing that is used with NAFs')
    return parser.parse_args()


def rm_tree(path: pathlib.Path) -> None:
    """ Clear specified directory and all subdirectories & files

    :param path: pathlib object for path that fill be deleted
    """
    path_obj = pathlib.Path(path)
    if not path_obj.exists():
        return
    for child in path_obj.glob('*'):
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    path.rmdir()


def save_coordinates(source: np.array, listener: np.array, fs: int, audio_length: int, path: str) -> None:  # todo: update for moving sources/listeners (?)
    """ Save the required coordinate and quaternion data for each audio to make them work as input data to the machine learning method

    (needed addition for some models: proper quaternions when directivity is used)
    :param source: source location (x, y, z), currently stays the same
    :param listener: listener location (x, y, z), currently stays the same
    :param fs: sample rate of the audio, used for generating coordinate data at fs/400 Hz
    :param audio_length: length of the audio, used for generating coordinate data at fs/400 Hz
    :param path: current subject folder under dataset
    :return: none
    """
    points = int(audio_length / fs * (fs/400))
    source_file = open(f'{path}/tx_positions.txt', 'a')
    listener_file = open(f'{path}/rx_positions.txt', 'a')
    for i in range(points):
        source_file.write(f'{source[0]} {source[1]} {source[2]}\n')  # add quaternions later, e.g., 1.0 0.0 0.0 0.0
        listener_file.write(f'{listener[0]} {listener[1]} {listener[2]}\n')
    source_file.close()
    listener_file.close()


def write_coordinate_metadata(grid: np.array, heights: list, room: list, room_name: str, save_path: str) -> None:
    """ Save the index and coordinates for each unique point in the grid as metadate, 0 1.0, 1.0, 1.5 etc.,
    as well as min/max coordinates in the room

    :param grid: x-y coordinate grid
    :param heights: heights of the sources and listeners, currently only uses one value
    :param room: room size
    :param room_name: name of the room for naming files and folders
    :param save_path: path to save the text file
    :return:
    """
    paths = [f'{save_path}/replica/{room_name}', f'{save_path}/minmax']
    for path in paths:
        check_and_create_dir(path)
    if pathlib.Path(f'{paths[0]}/points.txt').is_file():
        pathlib.Path(f'{paths[0]}/points.txt').unlink()
    with open(f'{paths[0]}/points.txt', 'a+') as f:
        for i, point in enumerate(grid):
            f.write(f'{i} {point[0]} {point[1]} {heights[0]}\n')
    with open(f'{paths[1]}/{room_name}_minmax.pkl', 'wb') as f:
        pickle.dump((np.array([0.0, 0.0, 0.0]), np.array(room)), f)


def write_split_metadata(grid: np.array, room_name: str, save_path: str) -> None:
    """ Create and store train/test split for point pairs that can be used as metadata for NAFs

    :param grid: x-y coordinate grid
    :param room_name: name of the room for naming files and folders
    :param save_path: path to save the pickle file
    :return:
    """
    save_path = f'{save_path}/train_test_split'
    check_and_create_dir(save_path)
    points = grid.shape[0]
    data = []
    for i in range(points):
        for j in range(points):
            if i == j:
                continue
            data.append(f'{i}_{j}')
    np.random.shuffle(data)
    pairs = points * (points - 1)
    train, test = int(np.floor(pairs * 0.9)), int(np.ceil(pairs * 0.1))
    split = [{0: data[:train]}, {0: data[train:train+test]}]
    with open(f'{save_path}/{room_name}_complete.pkl', 'wb') as f:
        pickle.dump(split, f)


def main():
    # Load arguments and create grid
    global RIRS
    args = parse_input_args()
    print(f'Generating SH RIR audio dataset in a room of size {args.room[0]}m*{args.room[1]}m*{args.room[2]}m with {args.grid[0]}x{args.grid[1]} grid and SH order {args.order}')
    grid = create_grid(np.array(args.grid), args.wall_gap, np.array(args.room))

    # Get dataset and save paths, load existing RIRs if possible
    parent_dir = str(pathlib.Path.cwd().parent)
    audio_data_path = f'{parent_dir}/{args.dataset_path}'
    save_path = f'{parent_dir}/{args.save_path}/ambisonics_{args.order}_{args.grid[0]}x{args.grid[1]}'
    rir_path = f'{parent_dir}/{args.save_path}/rirs/ambisonics_{args.order}/room_{args.room[0]}x{args.room[1]}x{args.room[2]}/grid_{args.grid[0]}x{args.grid[1]}/'
    metadata_path = f'{parent_dir}/{args.naf_path}/ambisonics_{args.order}_{args.grid[0]}x{args.grid[1]}'

    if pathlib.Path(f'{rir_path}/rirs.pickle').is_file():
        answer = input(f'Existing RIR file found at \'{rir_path}\', use stored RIRs for faster generation (y/n)? ')
        if answer.lower() in ['y', 'yes']:
            with open(f'{rir_path}/rirs.pickle', 'rb') as f:
                RIRS = pickle.load(f)
                print('Loaded existing RIR data')
    if pathlib.Path(save_path).is_dir():
        answer = input(f'Existing files found, all sub-folders/files in \'{save_path}\' will be deleted (ok/quit) ')
        if answer.lower() in ['o', 'ok', 'y', 'yes']:
            rm_tree(pathlib.Path(save_path))
            print('Old data cleared')
        else:
            print('Exiting; old data not deleted, try with different parameters or allow deletion to continue')
            sys.exit()

    # Store extra metadata
    if not args.skip_coord_write:
        write_coordinate_metadata(grid, args.heights, args.room, args.room_name, metadata_path)
    if not args.skip_split_write:
        write_split_metadata(grid, args.room_name, metadata_path)

    # Create dataset divided into trainset and testset
    audio_paths = get_audio_paths(f'{audio_data_path}/train_data.csv')
    generate_rir_audio_sh(grid, f'{save_path}/trainset', audio_paths, args=args)
    # audio_paths = get_audio_paths(f'{audio_data_path}/test_data.csv')
    # grid = create_grid(np.array([8, 4]), args.wall_gap, np.array(args.room))  # generate testset differently (smaller, different points?) (trainset is divided for NAFs)
    # generate_rir_audio_sh(grid, f'{save_path}/testset', audio_paths, args=args)

    # Save calculated RIRs
    if not args.skip_rir_write:  # might technically need even more specific file names
        check_and_create_dir(rir_path)
        with open(f'{rir_path}/rirs.pickle', 'wb') as f:
            pickle.dump(RIRS, f)


if __name__ == '__main__':
    main()
