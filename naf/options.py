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

    def check_paths(self):
        """
        Create default save/dataset paths based on grid size and ambisonics order,
        currently kind of stupid
        """
        save = f'ambisonics_{self.opt.order}_{self.opt.grid}'
        
        if self.opt.save_loc is None:
            self.opt.save_loc = f'./test_results/{save}'
        if self.opt.coor_base is None:
            self.opt.coor_base = f'./metadata/{save}/replica'
        if self.opt.spec_base is None:
            self.opt.spec_base = f'./metadata/{save}/magnitudes'
        if self.opt.phase_base is None:
            self.opt.phase_base = f'./metadata/{save}/phases'
        if self.opt.mean_std_base is None:
            self.opt.mean_std_base = f'./metadata/{save}/mean_std'
        if self.opt.minmax_base is None:
            self.opt.minmax_base = f'./metadata/{save}/minmax'
        if self.opt.wav_base is None:
            self.opt.wav_base = f'../../data/generated/ambisonics_{self.opt.order}_{self.opt.grid}'
        if self.opt.split_loc is None:
            self.opt.split_loc = f'./metadata/{save}/train_test_split/'
        if self.opt.wav_out is None:
            self.opt.wav_out = f'./out/{save}'

    def initialize(self):
        parser = self.parser
        parser.add_argument('--apt', default='test_1', choices=['test_1'], type=str)
        parser.add_argument('--grid', default='10x10', type=str)
        parser.add_argument('--order', default='0')
        parser.add_argument('--exp_name', default='{}')
        parser.add_argument('--save_loc', type=str)
        parser.add_argument('--wav_out', type=str)

        # dataset arguments, if not given, default ones are used (see check_paths())
        parser.add_argument('--coor_base', type=str)  # Location of the training index to coordinate mapping
        parser.add_argument('--spec_base', type=str)
        parser.add_argument('--phase_base', type=str)
        parser.add_argument('--mean_std_base', type=str)
        parser.add_argument('--minmax_base', type=str)
        parser.add_argument('--wav_base', type=str)
        parser.add_argument('--split_loc', type=str)

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
        parser.add_argument('--inference_loc', default='inference_out', type=str) # os.path.join(save_loc, inference_loc), where to cache inference results
        parser.add_argument('--gt_has_phase', default=0, type=bool_flag)  # image2reverb does not use gt phase for their GT when computing T60 error, and instead use random phase. If we use GT waveform (instead of randomizing the phase, we get lower T60 error)
        parser.add_argument('--emitter_loc', default=[1.0, 1.0], type=list_float_flag)

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.check_paths()
        self.opt.max_len = {'test_1': 28}  # Calculated when generating the dataset
        torch.manual_seed(0)
        np.random.seed(0)
        args = vars(self.opt)
        print('Arguments for the current run:')
        for k, v in args.items():
            print('    %s: %s' % (str(k), str(v)))
        print()
        return self.opt
