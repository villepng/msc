import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle

from scipy.fft import fft, ifft
from scipy.io import wavfile
from scipy.signal import deconvolve


# todo: ambisonic orders
def compare_rirs(rirs, gt_path, estimate_path, comparisons):
    subjects = len(glob.glob(f'{gt_path}/*'))
    comparisons = min(comparisons, subjects)
    for i, rir in enumerate(rirs.values()):
        if i + 1 > comparisons:
            break  # todo: make whole loop less stupid
        _, mono = wavfile.read(f'{gt_path}/subject{i + 1}/mono.wav')
        _, reverb = wavfile.read(f'{gt_path}/subject{i + 1}/ambisonic.wav')
        _, estimate = wavfile.read(f'{estimate_path}/subject{i + 1}.wav')
        # fig, axs = plt.subplots(3)
        # axs[0].plot(mono)
        # axs[0].set_title('mono')
        # axs[1].plot(reverb)
        # axs[1].set_title('gt')
        # axs[2].plot(estimate)
        # axs[2].set_title('estimate')
        # plt.show()

        # N = int(2 ** (np.ceil(np.log2(len(reverb) + len(mono)))))
        test2 = ifft(fft(reverb) / fft(mono))
        estimate_rir_normal = ifft(fft(estimate) / fft(mono))
        estimate_rir_cut = ifft(fft(estimate[2000:len(estimate)]) / fft(mono[2000:len(mono)]))
        test, _ = deconvolve(reverb, rir.squeeze())

        fig, axs = plt.subplots(4)
        axs[0].plot(rir)
        axs[0].set_title('Ground-truth RIR')
        axs[1].plot(test2[:len(rir)].real)
        axs[1].set_title('RIR calculated from ground-truth audio')
        axs[2].plot(estimate_rir_normal[:len(rir)].real)
        axs[2].set_title('Estimated RIR (full signal)')
        axs[3].plot(estimate_rir_cut[:len(rir)].real)
        axs[3].set_title('Estimated RIR (error at start cut)')
        plt.show()
        # print(np.sum(rir - estimate_rir) ** 1)


def parse_input_args():
    parser = argparse.ArgumentParser(description='calculate (and visualize ?) errors between ground-truth and estimated RIRs, assumes both are on the same grid')
    parser.add_argument('-r', '--rir_path', type=str, help='path to ground-truth RIRs')  # currently picled, todo: wav files/option for different types
    parser.add_argument('-g', '--gt_path', type=str, help='path to mono and ground-truth reverberant audio')
    parser.add_argument('-e', '--estimate_path', type=str, help='path to audio files generated with estimated RIRs')  # todo: "shorten" paths?
    parser.add_argument('-o', '--order', type=int, help='ambisonics order')
    parser.add_argument('--grid', nargs=2, type=int, help='grid size [x_n, y_n]')  # todo: could use gt path to get this, maybe safer this way
    parser.add_argument('-n', '--n_comparisons', default=10, type=int, help='number of rirs compared')  # tmp, depending on test set size could also compare all
    return parser.parse_args()


def main():
    args = parse_input_args()
    with open(f'{args.rir_path}', 'rb') as f:
        rirs = pickle.load(f)
    compare_rirs(rirs, args.gt_path, args.estimate_path, args.n_comparisons)


if __name__ == '__main__':
    main()
