import argparse
import numpy as np
import torch


def bool_flag(s):
    if s == '1':
        return True
    elif s == '0':
        return False
    msg = 'Invalid value "%s" for bool flag (should be 0 or 1)'
    raise ValueError(msg % s)


def list_float_flag(s):
    return [float(_) for _ in list(s)]


class Options:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        grid = '10x10'
        order = 0
        save = f'mono{grid}'  # ambisonic_1 etc. to change metadata folders faster
        parser = self.parser
        parser.add_argument('--save_loc', default=f'./test_results/{save}', type=str)
        parser.add_argument('--apt', default='test_1', choices=['test_1'], type=str)
        parser.add_argument('--exp_name', default='{}')

        # dataset arguments
        parser.add_argument('--coor_base', default=f'./metadata/{save}/replica', type=str)  # Location of the training index to coordinate mapping
        parser.add_argument('--spec_base', default=f'./metadata/{save}/magnitudes', type=str)
        parser.add_argument('--phase_base', default=f'./metadata/{save}/phases', type=str)
        parser.add_argument('--mean_std_base', default=f'./metadata/{save}/mean_std', type=str)
        parser.add_argument('--minmax_base', default=f'./metadata/{save}/minmax', type=str)
        parser.add_argument('--wav_base', default=f'../../data/generated/rir_ambisonics_order_{order}_{grid}', type=str)
        parser.add_argument('--split_loc', default=f'./metadata/{save}/train_test_split/', type=str)

        # training arguments
        parser.add_argument('--gpus', default=1, type=int)
        parser.add_argument('--epochs', default=200, type=int)
        parser.add_argument('--resume', default=0, type=bool_flag)  # Load weights or not from latest checkpoint
        parser.add_argument('--batch_size', default=20, type=int)
        parser.add_argument('--reg_eps', default=1e-1, type=float)  # Noise to regularize positions
        parser.add_argument('--pixel_count', default=2000, type=int)  # Noise to regularize positions
        parser.add_argument('--lr_init', default=5e-4, type=float)  # Starting learning rate
        parser.add_argument('--lr_decay', default=1e-1, type=float)  # Learning rate decay rate

        # network arguments
        parser.add_argument('--layers', default=8, type=int)  # Number of layers in the network
        parser.add_argument('--grid_gap', default=0.25, type=float)  # How far are the grid points spaced
        parser.add_argument('--bandwith_init', default=0.25, type=float)  # Initial bandwidth of the grid
        parser.add_argument('--features', default=512, type=int)  # Number of neurons in the network for each layer
        parser.add_argument('--grid_features', default=64, type=int)  # Number of neurons in the grid
        parser.add_argument('--position_float', default=0.1, type=float)  # Amount the position of each grid cell can float (up or down)
        parser.add_argument('--min_bandwidth', default=0.1, type=float)  # Minimum bandwidth for clipping
        parser.add_argument('--max_bandwidth', default=0.5, type=float)  # Maximum bandwidth for clipping
        parser.add_argument('--num_freqs', default=10, type=int)  # Number of frequency for sin/cos

        # testing arguments
        parser.add_argument('--wav_out', default=f'./out/{save}', type=str)
        parser.add_argument('--inference_loc', default='inference_out', type=str) # os.path.join(save_loc, inference_loc), where to cache inference results
        parser.add_argument('--gt_has_phase', default=0, type=bool_flag)  # image2reverb does not use gt phase for their GT when computing T60 error, and instead use random phase. If we use GT waveform (instead of randomizing the phase, we get lower T60 error)
        parser.add_argument('--emitter_loc', default=[1.0, 1.0], type=list_float_flag)

    def parse(self):
        # initialize parser
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.max_len = {'test_1': 28}  # Calculated when generating the dataset
        torch.manual_seed(0)
        np.random.seed(0)
        args = vars(self.opt)
        print('Arguments for the current run:')
        for k, v in args.items():
            print('    %s: %s' % (str(k), str(v)))
        print()
        return self.opt
