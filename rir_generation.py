import csv
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import pyroomacoustics as pra

from masp import shoebox_room_sim as srs
from scipy.io import wavfile
from scipy.signal import fftconvolve


def create_grid(x_points: int = 100, y_points: int = 100, wall_gaps: np.array = np.array([1.0, 1.0]),
                room_dim: np.array = np.array([10.0, 6.0, 3.0])) -> np.array:
    """ Create a grid of (x, y) coordinates with specified amount of points (x_points * y_points)

    :param x_points: points in x-axis
    :param y_points: points in y-axis
    :param wall_gaps: distance of the first point from the wall in x and y direction
    :param room_dim: room size x*y*z, where z is height
    :return: numpy array with x,y coordinate pairs

    """
    x = np.linspace(wall_gaps[0], room_dim[0] - wall_gaps[0], x_points)
    y = np.linspace(wall_gaps[1], room_dim[1] - wall_gaps[1], y_points)
    xx, yy = np.meshgrid(x, y)

    return np.vstack([xx.ravel(), yy.ravel()]).T


def generate_rir_audio_sh(points: np.array, save_path: str, audio_paths: np.array, heights: np.array = np.array([1.5, 1.5]), 
                          room: np.array = np.array([10.0, 6.0, 3.0]), rt60: np.array = np.array([0.2]), order: np.array = np.array([2])) -> None:
    """ Apply spherical harmonics RIR for specified audio at specified points; 
    for each point in the grid RIR applied audio at every other point is generated

    :param points: coordinate grid (x,y) in the room where RIR is calculated
    :param save_path: folder where to save the created audios
    :param audio_paths: array with paths to audio dataset wav files
    :param heights: array with source height and listener height, heights stay constant for all points
    :param room: room size x*y*z, where z is height
    :param rt60: reverberation time
    :param order: ambisonics order used
    :return: None

    """
    components = (order[0] + 1) ** 2
    nBands = len(rt60)
    # todo: check the variables below
    band_centerfreqs = np.empty(nBands)
    band_centerfreqs[0] = 1000
    # Absorption for approximately achieving the RT60 above - row per band
    abs_wall = srs.find_abs_coeffs_from_rt(room, rt60)[0]  # todo: update absorptions
    # Critical distance for the room (where reverberation = direct sound)
    _, d_critical, _ = srs.room_stats(room, abs_wall)

    audio_index = 0
    data_index = 0
    for i, src_pos in enumerate(points):
        for j, lstn_pos in enumerate(points):
            fs, audio_anechoic = wavfile.read(audio_paths[audio_index])
            if j == i:
                continue  # skip if source and listener are at the same point

            source = np.array([[src_pos[0], src_pos[1], heights[0]]])
            receiver = np.array([[lstn_pos[0], lstn_pos[1], heights[1]]])
        
            # Echogram
            maxlim = 1.5 # just stop if the echogram goes beyond that time (or just set it to max(rt60))
            limits = np.minimum(rt60, maxlim)
            abs_echograms = srs.compute_echograms_sh(room, source, receiver, abs_wall, limits, order)  # todo: check that coordinate systems match
            
            # In this case all the information (e.g. SH directivities) are already
            # encoded in the echograms, hence they are rendered directly to discrete RIRs
            sh_rirs = srs.render_rirs_sh(abs_echograms, band_centerfreqs, fs).squeeze()
            if order[0] == 0:
                sh_rirs = sh_rirs.reshape(len(sh_rirs), 1)
            sh_rirs = sh_rirs * np.sqrt(4*np.pi) * get_sn3d_norm_coefficients(order=order[0])
            #plt.figure()
            #plt.plot(sh_rirs)
            #plt.show()
            audio_length = len(audio_anechoic)

            reverberant_signal = np.zeros((audio_length, components))
            pathlib.Path(f'{save_path}/subject{data_index + 1}').mkdir(parents=True)
            wavfile.write(f'{save_path}/subject{data_index + 1}/mono.wav', fs, audio_anechoic.astype(np.int16))
            for k in range(components):
                reverberant_signal[:, k] = fftconvolve(audio_anechoic, sh_rirs[:, k].squeeze())[:audio_length]
                wavfile.write(f'{save_path}/subject{data_index + 1}/ambisonic_{k}.wav', fs, reverberant_signal[:, k].astype(np.int16))

            save_coordinates(source=np.array([src_pos[0], src_pos[1], heights[0]]), listener=np.array([lstn_pos[0], lstn_pos[1], heights[1]]),
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
                path_start = '/'.join(path.split('/')[:-1])  # get path to timit folder from csv path
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

def rm_tree(path: pathlib.Path) -> None:
    """ Clear specified directory

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

    todo: proper quaternions when directivity is used, testing with no quaternions for the first step
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


# todo: startup args? (room size, grid, save and data folders etc.)
def main():
    room = np.array([10.0, 6.0, 3.0])
    grid = create_grid(x_points=2, y_points=2, wall_gaps=np.array([1.0, 1.0]), room_dim=room)
    heights = np.array([1.5, 1.5])  # source, listener
    rt60 = np.array([0.2])
    order = np.array([1])

    parent_dir = str(pathlib.Path.cwd().parent)
    audio_data_path = f'{parent_dir}/data/timit'  # using TIMIT dataset for now
    save_path = f'{parent_dir}/data/generated/rir_ambisonics_order_{order}'
    rm_tree(pathlib.Path(save_path))  # clear old files

    # train data in save path under trainset folder
    audio_paths = get_audio_paths(f'{audio_data_path}/train_data.csv')
    generate_rir_audio_sh(grid, f'{save_path}/trainset', audio_paths, heights, room, rt60, order)
    # test data in save path under testset folder
    audio_paths = get_audio_paths(f'{audio_data_path}/test_data.csv')
    generate_rir_audio_sh(grid, f'{save_path}/testset', audio_paths, heights, room, rt60, order)


if __name__ == '__main__':
    main()
