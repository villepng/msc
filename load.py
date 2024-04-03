import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pickle

from scipy.io import wavfile

from naf.utils import plot_wave_ambi
from rir_generation import write_coordinate_metadata


def check_and_create_dir(path: str) -> None:
    """ Create directory if it doesn't exist

    :param path: path string (or object?)
    :return:
    """
    if not pathlib.Path(path).exists():
        pathlib.Path(path).mkdir(parents=True)


def create_grid(points: np.array, wall_gap: np.array, room_dim: np.array) -> np.array:
    """ Create a grid of (x, y) coordinates with specified amount of points (x_points * y_points)

    :param points: grid points in x and y directions, total points will be x*y
    :param wall_gap: minimum distance between walls and grid points
    :param room_dim: room size x*y*z, where z is height
    :return: numpy array with x,y coordinate pairs

    """
    x = np.linspace(wall_gap[0], room_dim[0] - wall_gap[1], points[0])
    y = np.linspace(wall_gap[2], room_dim[1] - wall_gap[3], points[1])
    xx, yy = np.meshgrid(x, y)

    return np.vstack([xx.ravel(), yy.ravel()]).T


def find_index(grid, point):
    """ Find index of point in grid

    :param grid:
    :param point:
    :return:
    """
    for i, point_i in enumerate(grid):
        if np.array_equal(point_i, point):
            return i
    raise LookupError(f'point {point} not found in the given grid of points')


def write_split_metadata(grid: np.array, room_name: str, save_path: str) -> None:
    """ Create and store train/test split for point pairs that can be used as metadata for NAFs

    :param grid: x-y coordinate grid
    :param room_name: name of the room for naming files and folders
    :param save_path: path to save the pickle file
    :return:
    """
    rng = np.random.default_rng(0)
    save_path = f'{save_path}/train_test_split'
    check_and_create_dir(save_path)
    points = grid.shape[0]
    data = []
    for i in range(points):
        for j in range(points):
            if i == j:
                continue
            data.append(f'{i}_{j}')
    rng.shuffle(data)
    pairs = points * (points - 1)
    train, test = int(np.floor(pairs * 0.9)), int(np.ceil(pairs * 0.1))
    split = [{0: data[:train]}, {0: data[train:train+test]}]
    with open(f'{save_path}/{room_name}_complete.pkl', 'wb') as f:
        pickle.dump(split, f)


def main():
    room = 'real_1'
    save_path = 'naf/metadata/ambisonics_1_10x13'
    check_and_create_dir(save_path)
    grid = create_grid([10, 13], [2.0, 1.0, 1.5, 1.5], [7.5, 9, 3.5])
    grid = np.concatenate((grid, np.array([[0.5, 4.5]])))  # tmp speaker position
    write_coordinate_metadata(grid, [1.5, 1.5], [8, 8, 3.5], room, save_path)
    write_split_metadata(grid, room, save_path)

    folder = '../data/isophonics/'
    rirs = {}
    for chn in ['W', 'Y', 'Z', 'X']:
        for file in pathlib.Path(f'{folder}/classroom{chn}/{chn}/').glob('*.wav'):
            fs, rir = wavfile.read(file)
            # plt.plot(rir)
            # plt.show()
            info = file.parts[-1][1:6]
            x, y = float(info[-2:]) / 10 + 2.0, float(info[:2]) / 10 + 1.5  # flip to match with original coordinate system and add wall gaps
            i = find_index(grid, np.array([x, y]))
            if f'130-{i}' not in rirs:
                rirs.update({f'130-{i}': []})
            rirs[f'130-{i}'].append(rir)

    # reformat to match original data
    for key, item in rirs.items():
        rirs[key] = np.array(item).T
    rir_save_path = f'../data/generated/rirs/ambisonics_1/room_7.5x9.0x3.5/grid10x13'
    check_and_create_dir(rir_save_path)
    with open(f'{rir_save_path}/rirs.pickle', 'wb') as f:
        pickle.dump(rirs, f)

    plot_wave_ambi(rirs['130-0'].T, rirs['130-0'].T, 'test', fs)
    # plt.plot(rirs['130-0'][0])
    # plt.show()


if __name__ == '__main__':
    main()

