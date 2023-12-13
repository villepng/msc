import argparse
import numpy as np
import pathlib
import pickle


def parse_input_args():
    parser = argparse.ArgumentParser(description='calculate (and visualize ?) errors between ground-truth and estimated RIRs, assumes both are on the same grid')
    parser.add_argument('-t', '--gt_path', type=str, help='path to ground-truth RIRs')  # currently picled, todo: wav files/option for different types
    parser.add_argument('-s', '--save_path', type=str, help='path to audio files generated with estimated RIRs')  # todo: "shorten" paths?
    parser.add_argument('-o', '--order', type=int, help='ambisonics order')
    parser.add_argument('-g', '--grid', nargs=2, type=int, help='grid size [x_n, y_n]')  # todo: could use gt path to get this, maybe safer this way
    parser.add_argument('-n', '--n_comparisons', default=10, type=int, help='number of rirs compared')  # tmp, depending on test set size could also compare all
    return parser.parse_args()


def main():
    args = parse_input_args()
    with open(f'{args.gt_path}', 'rb') as f:
        print('Loading existing RIR data')
        rirs = pickle.load(f)


if __name__ == '__main__':
    main()
