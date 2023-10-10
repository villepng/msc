import numpy as np
import matplotlib.pyplot as plt
import pathlib
import pyroomacoustics as pra

from scipy.io import wavfile


def check_path(path: str) -> None:
    """ Check if path exists, create it if not and delete existing wav files it if it does
    :param path: path to check

    :return: None

    """
    if pathlib.Path(path).is_dir():
        [f.unlink() for f in pathlib.Path(path).glob('*.wav') if f.is_file()]
    else:
        pathlib.Path(path).mkdir(parents=True)


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


def generate_rir_audio(points: np.array, e_absorption: float, max_order: int, path: str, source_height: float = 1.5,
                       mic_height: float = 1.5, room_dim: np.array = np.array([10.0, 6.0, 3.0])) -> None:
    """ Apply RIR for specified audio at specified points, for each point
    in the grid RIR applied audio at every other point is generated
    :param points: coordinate grid (x,y) in the room where RIR is calculated
    :param e_absorption: parameter for RIR calculation
    :param max_order: parameter for RIR calculation
    :param path: folder where to save the created audios
    :param source_height: height stays constant for all points, should be above 0 and smaller than z in room_size
    :param mic_height: height stays constant for all points, should be above 0 and smaller than z in room_size
    :param room_dim: room size x*y*z, where z is height

    :return: None

    """
    for audio in ['download.wav']:
        fs, audio_anechoic = wavfile.read(audio)
        for i, point_src in enumerate(points):
            # todo: separate folder ('subject') for each snippet, or combine all x*y*(x*y-1) into one?
            for j, point_mic in enumerate(points):
                if j == i:
                    continue
                room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order)
                room.add_source([point_src[0], point_src[1], source_height], signal=audio_anechoic, delay=0.5)
                # print(f'Source: ({point_src[0]}, {point_src[1]}); Listener: ({point_mic[0]}, {point_mic[1]})')
                room.add_microphone([point_mic[0], point_mic[1], mic_height])
                room.simulate()
                # room.plot_rir()
                # plt.show()
                room.mic_array.to_wav(f'{path}{audio.split(".")[0]}_rir_{i}-{j}.wav', norm=True, bitdepth=np.int16)
                # save_coordinates(source=np.array([point_src[0], point_src[1], source_height]), listener=np.array([point_mic[0], point_mic[1], mic_height]))
                if i == 0 and j == 1:
                    save_coordinates(source=np.array([point_src[0], point_src[1], source_height]), listener=np.array([point_mic[0], point_mic[1], mic_height]),
                                     fs=fs, audio_length=len(audio_anechoic))


def save_coordinates(source: np.array, listener: np.array, fs: int, audio_length: int) -> None:  # todo: update for moving sources/listeners (?)
    """ Save the required coordinate and quaternion data for each audio to make them work as input data to the machine learning method
    todo: proper quaternions when directivity is used
    :param source: source location (x, y, z), currently stays the same
    :param listener: listener location (x, y, z), currently stays the same
    :param fs: sample rate of the audio, used for generating coordinate data at 120 Hz
    :param audio_length: length of the audio, used for generating coordinate data at 120 Hz

    :return: none
    """
    points = int(audio_length // fs * 120)
    # todo: separate folder for each audio and coordinate data thingy
    source_file = open('tx_positions.txt', 'a')  # todo: check if exists or do already at folder level?
    listener_file = open('rx_positions.txt', 'a')
    for i in range(points):
        source_file.write(f'{source[0]}, {source[1]}, {source[2]}, 1.0, 0.0, 0.0, 0.0\n')
        listener_file.write(f'{listener[0]}, {listener[1]}, {listener[2]}, 1.0, 0.0, 0.0, 0.0\n')
    source_file.close()
    listener_file.close()


# todo: startup args?
def main():
    room_size = [10.0, 6.0, 3.0]
    grid = create_grid(x_points=2, y_points=2, wall_gaps=np.array([0.01, 0.01]), room_dim=np.array(room_size))
    reverb_time = 0.5  # todo: use 0.2 like in the original work's baseline?
    e_absorption, max_order = pra.inverse_sabine(reverb_time, room_size)
    save_path = 'D:\\Python\\tmp\\rir\\'  # todo: separate folder for each audio probably
    check_path(path=save_path)
    generate_rir_audio(points=grid, e_absorption=e_absorption, max_order=max_order, path=save_path,
                       source_height=1.5, mic_height=1.5, room_dim=np.array(room_size))


if __name__ == '__main__':
    main()
