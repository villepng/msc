import csv
import numpy as np
import pathlib
import spaudiopy as spa  # currently requires python versions >=3.6 but < 3.12

from scipy.io import wavfile


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
    # todo: use angle to get the correct hrir and return it, angles might need tweaking (what is front etc.)



def main():
    parent_dir = str(pathlib.Path.cwd().parent)
    order = 1

    for i in range(1, 13):  # todo: grid points as a parameter (starup args?)
        audio_data_path = f'{parent_dir}/data/generated/rir_ambisonics_order_{order}/testset/subject{i}'
        fs, ambisonic = wavfile.read(f'{audio_data_path}/ambisonic.wav')

        hrir, _ = spa.io.sofa_to_sh(f'{parent_dir}/data/irs etc/mit_kemar_normal_pinna.sofa', order)
        binaural = spa.decoder.sh2bin(ambisonic.T, hrir)
    
        wavfile.write(f'{parent_dir}/data/out/binaural_{i}.wav', fs, binaural.astype(np.int16).T)  # todo: create and name folder


if __name__ == '__main__':
    main()
