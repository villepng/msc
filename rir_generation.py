import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import pathlib

from masp import shoebox_room_sim as srs
from scipy.io import wavfile
from scipy.signal import fftconvolve


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


def generate_rir_audio_sh(points: np.array, save_path: str, audio_paths: np.array, heights: np.array, 
                          room: np.array, rt60: float, order: int) -> None:
    """ Apply spherical harmonics RIR for specified audio at specified points; 
    for each point in the grid RIR applied audio at every other point is generated
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
    :return: None

    """
    rt60 = np.array([rt60])
    components = (order + 1) ** 2
    nBands = len(rt60)
    # todo: check the variables below
    band_centerfreqs = np.empty(nBands)
    band_centerfreqs[0] = 1000
    # Absorption for approximately achieving the RT60 above - row per band
    abs_wall = srs.find_abs_coeffs_from_rt(room, rt60)[0]  # todo: update absorptions
    # Critical distance for the room (where reverberation = direct sound, check and warn if src->lstn is bigger?)
    _, d_critical, _ = srs.room_stats(room, abs_wall)

    audio_index = 0
    data_index = 0
    for i, src_pos in enumerate(points):
        for j, recv_pos in enumerate(points):
            fs, audio_anechoic = wavfile.read(audio_paths[audio_index])
            if j == i:
                continue  # skip if source and listener are at the same point

            source = np.array([[src_pos[0], src_pos[1], heights[0]]])
            receiver = np.array([[recv_pos[0], recv_pos[1], heights[1]]])
        
            maxlim = 1.5 # just stop if the echogram goes beyond that time (or just set it to max(rt60))
            limits = np.minimum(rt60, maxlim)
            abs_echograms = srs.compute_echograms_sh(room, source, receiver, abs_wall, limits, order)
            
            # In this case all the information (e.g. SH directivities) are already
            # encoded in the echograms, hence they are rendered directly to discrete RIRs
            sh_rirs = srs.render_rirs_sh(abs_echograms, band_centerfreqs, fs).squeeze()
            if order == 0:
                sh_rirs = sh_rirs.reshape(len(sh_rirs), 1)
            sh_rirs = sh_rirs * np.sqrt(4*np.pi) * get_sn3d_norm_coefficients(order)
            #plt.figure()
            #plt.plot(sh_rirs)
            #plt.show()
            audio_length = len(audio_anechoic)

            reverberant_signal = np.zeros((audio_length, components))
            pathlib.Path(f'{save_path}/subject{data_index + 1}').mkdir(parents=True)
            wavfile.write(f'{save_path}/subject{data_index + 1}/mono.wav', fs, audio_anechoic.astype(np.int16))
            for k in range(components):
                reverberant_signal[:, k] = fftconvolve(audio_anechoic, sh_rirs[:, k].squeeze())[:audio_length]
            wavfile.write(f'{save_path}/subject{data_index + 1}/ambisonic.wav', fs, reverberant_signal.astype(np.int16))

            save_coordinates(source=np.array([src_pos[0], src_pos[1], heights[0]]), listener=np.array([recv_pos[0], recv_pos[1], heights[1]]),
                             fs=fs, audio_length=audio_length, path=f'{save_path}/subject{data_index + 1}/')

            audio_index += 1
            data_index += 1
            if audio_index == len(audio_paths):  # go through all audio snippets using different one for each data point and start over if needed
                audio_index = 0


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

def parse_input_args():
    parser = argparse.ArgumentParser(description='Create reverberant audio dataset encoded into ambisonics with specified order')
    parser.add_argument('-d', '--dataset_path', default='data/timit', type=str, help='path to TIMIT dataset from current parent folder')  # todo: make paths "normal" (?)
    parser.add_argument('-s', '--save_path', default='data/generated', type=str, help='path (from current parent folder) where to save the generated dataset, \
                        will be saved in a folder named based on the ambisonics order')
    parser.add_argument('-r', '--room', nargs=3, default=[10.0, 6.0, 2.5], type=float, help='room size as (x y z)', metavar=('room_x', 'room_y', 'room_z'))
    parser.add_argument('-g', '--grid', nargs=2, default=[2, 2], type=int, help='grid points in each axis (x y)', metavar=('x_n', 'y_n'))  # todo: change to 100 later?
    parser.add_argument('-w', '--wall_gap', default=1.0, type=float, help='minimum gap between walls and grid points')
    parser.add_argument('--heights', nargs=2, default=[1.5, 1.5], type=float, help='heights for the source and the listener', metavar=('source_height', 'listener_height'))
    parser.add_argument('--rt60', default=0.2, type=float, help='reverberation time of the room')
    parser.add_argument('-o', '--order', default=1, type=int, help='ambisonics order')
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

    todo: proper quaternions when directivity is used, testing with no quaternions for the first step (possibility of only using angles?)
    :param source: source location (x, y, z), currently stays the same
    :param listener: listener location (x, y, z), currently stays the same
    :param fs: sample rate of the audio, used for generating coordinate data at fs/400 Hz
    :param audio_length: length of the audio, used for generating coordinate data at fs/400 Hz
    :param path: current subject folder under dataset
    :return: none
    """
    points = int(audio_length / fs * (fs/400))
    source_file = open(f'{path}tx_positions.txt', 'a')
    listener_file = open(f'{path}rx_positions.txt', 'a')
    for i in range(points):
        source_file.write(f'{source[0]} {source[1]} {source[2]}\n')  # add quaternions later, e.g., 1.0 0.0 0.0 0.0
        listener_file.write(f'{listener[0]} {listener[1]} {listener[2]}\n')
    source_file.close()
    listener_file.close()


# todo: fix type hints and function documentations
def main():
    args = parse_input_args()

    grid = create_grid(np.array(args.grid), args.wall_gap, np.array(args.room))

    parent_dir = str(pathlib.Path.cwd().parent)
    audio_data_path = f'{parent_dir}/{args.dataset_path}'
    save_path = f'{parent_dir}/{args.save_path}/rir_ambisonics_order_{args.order}'
    rm_tree(pathlib.Path(save_path))  # clear old files

    # train data in save path under trainset folder
    audio_paths = get_audio_paths(f'{audio_data_path}/train_data.csv')
    generate_rir_audio_sh(grid, f'{save_path}/trainset', audio_paths, np.array(args.heights), np.array(args.room), args.rt60, args.order)
    # test data in save path under testset folder
    audio_paths = get_audio_paths(f'{audio_data_path}/test_data.csv')
    generate_rir_audio_sh(grid, f'{save_path}/testset', audio_paths, np.array(args.heights), np.array(args.room), args.rt60, args.order)


if __name__ == '__main__':
    main()
