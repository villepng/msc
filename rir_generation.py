import csv
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import pyroomacoustics as pra

from scipy.io import wavfile


def create_grid(x_points: int = 100, y_points: int = 100, wall_gaps: np.array = np.array([0.01, 0.01]),
                room_dim: np.array = np.array([10.0, 6.0, 3.0])) -> np.array:
    """ Create a grid of (x, y) coordinates with specified size
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


def generate_rir_audio(points: np.array, materials: dict, max_order: int, data_path: str, save_path: str,
                       source_height: float = 1.5, mic_height: float = 1.5, room_dim: np.array = np.array([10.0, 6.0, 3.0])) -> None:
    """ Apply RIR for specified audio at specified points, for each point
    in the grid RIR applied audio at every other point is generated
    :param points: coordinate grid (x,y) in the room where RIR is calculated
    :param materials: room materials for RIR calculation
    :param max_order: parameter for RIR calculation
    :param data_path: folder where TIMIT dataset is saved
    :param save_path: folder where to save the created audios
    :param source_height: height stays constant for all points, should be above 0 and smaller than z in room_size
    :param mic_height: height stays constant for all points, should be above 0 and smaller than z in room_size
    :param room_dim: room size x*y*z, where z is height

    :return: None

    """
    audio_paths = get_audio_paths(data_path)
    audio_index = 0
    for i, point_src in enumerate(points):
        for j, point_mic in enumerate(points):
            fs, audio_anechoic = wavfile.read(audio_paths[audio_index])
            if j == i:
                continue
            room = pra.ShoeBox(room_dim, fs=fs, materials=materials, max_order=max_order)
            room.add_source([point_src[0], point_src[1], source_height], signal=audio_anechoic, delay=0.5)
            room.add_microphone([point_mic[0], point_mic[1], mic_height])
            room.simulate()
            # room.plot_rir()
            # plt.show()

            # tmp solution to match mono and rir mono lengths
            length_rir = len(room.mic_array.signals[0])
            mono = np.zeros([length_rir])
            mono[0:len(audio_anechoic)] = audio_anechoic

            pathlib.Path(f'{save_path}subject{audio_index + 1}').mkdir(parents=True, exist_ok=True)  # todo: make more sensible
            wavfile.write(f'{save_path}subject{audio_index + 1}\\mono.wav', fs, mono.astype(np.int16))
            room.mic_array.to_wav(f'{save_path}subject{audio_index + 1}\\binaural.wav', norm=True, bitdepth=np.int16)
            save_coordinates(source=np.array([point_src[0], point_src[1], source_height]), listener=np.array([point_mic[0], point_mic[1], mic_height]),
                             fs=fs, audio_length=length_rir, path=f'{save_path}subject{audio_index + 1}\\')

            audio_index += 1  # todo: needs fixing when used for naming
            if audio_index == len(audio_paths):
                audio_index = 0


def get_audio_paths(path: str) -> np.array:
    """ Use TIMIT dataset's train_data.csv to get paths to all files used for creating the training data
    :param path: path to TIMIT data
    :return: array with full paths to included wav files
    """
    paths = []
    with open(f'{path}train_data.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['is_converted_audio'] == 'TRUE':
                paths.append(f'{path}data\\{row["path_from_data_dir"]}')
    return np.array(paths)


def rm_tree(path: pathlib.Path) -> None:
    """ Clear specified directory
    :param path: pathlib object for path that fill be deleted
    """
    path_obj = pathlib.Path(path)
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
    # todo: separate folder for each audio and coordinate data thingy
    source_file = open(f'{path}tx_positions.txt', 'a')
    listener_file = open(f'{path}rx_positions.txt', 'a')
    for i in range(points):
        source_file.write(f'{source[0]} {source[1]} {source[2]}\n')  # add quaternions later, e.g., 1.0 0.0 0.0 0.0
        listener_file.write(f'{listener[0]} {listener[1]} {listener[2]}\n')
    source_file.close()
    listener_file.close()


# todo: startup args? (room size, grid, save and data folders etc.)
def main():
    room_size = [10.0, 6.0, 2.5]
    grid = create_grid(x_points=2, y_points=2, wall_gaps=np.array([1.0, 1.0]), room_dim=np.array(room_size))
    # reverb_time = 0.2
    # e_absorption, max_order = pra.inverse_sabine(reverb_time, room_size)
    materials = pra.make_materials(
        ceiling='wood_16mm',
        floor='carpet_soft_10mm',
        east='wood_16mm',
        west='wood_16mm',
        north='glass_window',
        south='wooden_door',
    )

    audio_data_path = 'D:\\Python\\timit\\'  # using TIMIT dataset for now
    save_path = 'D:\\Python\\tmp\\rir\\'
    path_obj = pathlib.Path(save_path)
    rm_tree(path_obj)  # clear old files
    generate_rir_audio(points=grid, materials=materials, max_order=8, data_path=audio_data_path,
                       save_path=save_path, source_height=1.5, mic_height=1.5, room_dim=np.array(room_size))


if __name__ == '__main__':
    main()
