import librosa
import matplotlib.pyplot as plt
import numpy as np
import pickle

from scipy.io import wavfile
from scipy.signal import fftconvolve


def fix_data(gt, sim, path):
    # normalise simulated rirs
    max_val = -np.inf
    for key, item in sim.items():
        if np.max(item) > max_val:
            max_val = np.max(item)
    for key, item in sim.items():
        sim[key] = np.array(item) / max_val

    # match rir sample rates and lengths
    for key, item in gt.items():
        rir_gt = librosa.resample(item, orig_sr=96000, target_sr=16000, axis=0)
        gt[key] = rir_gt[:len(sim['6-0'][:, 0]), :]

    # update to use real rirs at available points
    for key, item in sim.items():
        if '6-' in key:
            sim[key] = gt[key]

    with open(f'{path}/grid_10x13/rirs2.pickle', 'wb') as fg:
        pickle.dump(gt, fg)
    with open(f'{path}/grid_3x5/rirs2.pickle', 'wb') as fs:
        pickle.dump(sim, fs)


def main():
    root = '/worktmp/melandev/data/generated/rirs/ambisonics_1/room_7.5x9.0x3.5'
    with open(f'{root}/grid_10x13/rirs.pickle', 'rb') as f1:
        gt = pickle.load(f1)
    with open(f'{root}/grid_3x5/rirs.pickle', 'rb') as f2:
        sim = pickle.load(f2)
    fix_data(gt, sim, root)

    fs, mono = wavfile.read(f'../data/generated/ambisonics_1_20x10/trainset/subject1/mono.wav')

    rir_s = sim['6-0']
    rir_gt = gt['6-0']
    plt.figure()
    plt.plot(rir_s[:, 1])
    plt.figure()
    plt.plot(rir_gt[:, 1])
    plt.show()

    reverb = []
    reverb_gt = []
    for k in range(4):
        reverb.append(fftconvolve(mono, rir_s[:, k].squeeze()))
        reverb_gt.append(fftconvolve(mono, rir_gt[:, k].squeeze()))
    reverb, reverb_gt = np.array(reverb).T, np.array(reverb_gt).T
    wavfile.write(f'tmp/sim.wav', fs, reverb.astype(np.int16))
    wavfile.write(f'tmp/gt.wav', fs, reverb_gt.astype(np.int16))


if __name__ == '__main__':
    main()
