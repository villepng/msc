import argparse
import csv
import numpy as np
import pathlib
import spaudiopy as spa  # currently requires python versions >=3.6 but < 3.12

from scipy.io import wavfile

from rir_generation import rm_tree


# not needed currently, can be used to check angles
def get_hrir(data_path: str):
    """ Get HRIR at the correct angle based on the angle between the source
    and the receiver (position data in rx and tx files respectively)

    :param data_path: path to a subject's audio and position files
    """
    # Currently positions don't change, so reading the first line is enough to get the positions
    # todo: movement and quaternions, different heights (?)
    with open(f'{data_path}/tx_positions.txt', newline='') as source:
        reader = csv.reader(source, delimiter=' ')
        src_pos = list(map(float, next(reader)))
    with open(f'{data_path}/rx_positions.txt', newline='') as receiver:
        reader = csv.reader(receiver, delimiter=' ')
        rcv_pos = list(map(float, next(reader)))
    # Get angle [0, 2*pi[ so that "front" is 0 radians (facing the negative y direction), todo: update (quaternions etc.?)
    angle = np.arctan2(src_pos[1] - rcv_pos[1], src_pos[0] - rcv_pos[0])
    if angle < 0:
        angle = angle + 2 * np.pi
    print(f'source: {src_pos[0:2]}, receiver: {rcv_pos[0:2]}, angle: {angle * 180 / np.pi}')
    # todo: use angle to get the correct hrir and return it


def parse_input_args():
    parser = argparse.ArgumentParser(description='Convert ambisonics encoded audio into binaural audio (tmp version)')
    parser.add_argument('-d', '--data_path', default='data/generated', type=str, help='path to ambisonics dataset directory from current parent folder')
    parser.add_argument('-s', '--save_path', default='data/out', type=str, help='path (from current parent folder) where to save the binaural files, \
                        will be saved in a folder named based on the ambisonics order')
    parser.add_argument('-o', '--order', default=1, type=int, help='ambisonics order')
    parser.add_argument('--hrir', default='data/irs etc/mit_kemar_normal_pinna.sofa', type=str, help='path to the hrir (sofa file) to be used from current parent folder')
    return parser.parse_args()


def main():
    args = parse_input_args()
    order = args.order
    parent_dir = str(pathlib.Path.cwd().parent)
    save_path = f'{parent_dir}/{args.save_path}/order_{order}'

    data_path_obj = pathlib.Path(f'{parent_dir}/{args.data_path}/rir_ambisonics_order_{order}/testset')
    subjects = len([p for p in data_path_obj.iterdir() if p.is_dir()])

    rm_tree(pathlib.Path(save_path))  # clear old files
    pathlib.Path(save_path).mkdir(parents=True)

    hrir, _ = spa.io.sofa_to_sh(f'{parent_dir}/{args.hrir}', order)
    for i in range(1, subjects + 1):  # subject indexing starts from 1 due to how the ML method is set up
        audio_data_path = f'{str(data_path_obj)}/subject{i}'
        fs, ambisonic = wavfile.read(f'{audio_data_path}/ambisonic.wav')
        binaural = spa.decoder.sh2bin(ambisonic.T, hrir)
        wavfile.write(f'{save_path}/binaural_{i}.wav', fs, binaural.astype(np.int16).T)  # todo: create and name folder


if __name__ == '__main__':
    main()
