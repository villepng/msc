import numpy as np
import pathlib
import pickle

from scipy.io import wavfile


def main():
    data_path = '/worktmp/melandev/data/generated/rir_ambisonics_order_0_10x10'
    new_path = f'{data_path}_rir'
    rir_path = '/worktmp/melandev/data/generated/rirs/order_0/room_10.0x6.0x2.5/grid_10x10/rt60_0.2/rirs.pickle'
    with open(rir_path, 'rb') as f:
        rirs = pickle.load(f)
    fs = 16000

    for section in ['trainset', 'testset']:
        for subject, rir in enumerate(rirs.values()):
            path = f'{new_path}/{section}/subject{subject + 1}'
            pathlib.Path(path).mkdir(parents=True)
            wavfile.write(f'{path}/mono.wav', fs, rir.astype(np.int16))


if __name__ == '__main__':
    main()
