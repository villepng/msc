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


class Options:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser(description='Parameters for various scripts related to NAFs, if paths are not given, '
                                                          'default values for them are generated based on the ambisonics order and grid size')
        self.initialized = False

    def check_paths(self):
        """
        Create default save/dataset paths based on grid size and ambisonics order,
        currently kind of stupid
        """
        save = f'ambisonics_{self.opt.order}_{self.opt.grid}_rm'
        
        if self.opt.model_save_loc is None:
            self.opt.model_save_loc = f'./test_results/{save}'
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
            self.opt.wav_base = f'../../data/generated/{save}'
        if self.opt.early_base is None:
            self.opt.early_base = f'./metadata/{save}/early'
        if self.opt.split_loc is None:
            self.opt.split_loc = f'./metadata/{save}/train_test_split/'
        if self.opt.wav_loc is None:
            self.opt.wav_loc = f'./out/{save}/wav'
        if self.opt.metric_loc is None:
            self.opt.metric_loc = f'./out/{save}/metrics'

    def initialize(self):
        parser = self.parser
        parser.add_argument('--apt', default='test_1', choices=['test_1'], type=str)
        parser.add_argument('--grid', default='20x10', type=str)
        parser.add_argument('--order', default='1')
        parser.add_argument('--exp_name', default='{}')
        parser.add_argument('--model_save_loc', type=str, help='change this to test with a grid of points different from training')

        # dataset arguments, if not given, default ones are used (see check_paths())
        parser.add_argument('--coor_base', type=str, help='location of the training index to coordinate mapping')
        parser.add_argument('--spec_base', type=str)
        parser.add_argument('--phase_base', type=str)
        parser.add_argument('--mean_std_base', type=str)
        parser.add_argument('--minmax_base', type=str)
        parser.add_argument('--wav_base', type=str)
        parser.add_argument('--early_base', type=str)
        parser.add_argument('--split_loc', type=str)
        parser.add_argument('--n_fft', default=128, type=int)
        parser.add_argument('--hop_len', default=64, type=int)

        # training arguments
        parser.add_argument('--gpus', default=1, type=int)
        parser.add_argument('--epochs', default=200, type=int)
        parser.add_argument('--resume', default=0, type=bool_flag, help='load weights or not from latest checkpoint')
        parser.add_argument('--batch_size', default=20, type=int)
        parser.add_argument('--reg_eps', default=1e-1, type=float, help='noise to regularize positions')
        parser.add_argument('--pixel_count', default=2000, type=int, help='noise to regularize positions')
        parser.add_argument('--lr_init', default=5e-4, type=float, help='starting learning rate')
        parser.add_argument('--lr_decay', default=1e-1, type=float, help='learning rate decay rate')

        # network arguments
        parser.add_argument('--layers', default=8, type=int, help='number of layers in the network')
        parser.add_argument('--grid_gap', default=0.4, type=float, help='how far are the grid points spaced')
        parser.add_argument('--bandwith_init', default=0.4, type=float, help='initial bandwidth of the grid')
        parser.add_argument('--features', default=512, type=int, help='number of neurons in the network for each layer')
        parser.add_argument('--grid_features', default=64, type=int, help='number of neurons in the grid')
        parser.add_argument('--position_float', default=0.1, type=float, help='amount the position of each grid cell can float (up or down)')
        parser.add_argument('--min_bandwidth', default=0.1, type=float, help='minimum bandwidth for clipping')
        parser.add_argument('--max_bandwidth', default=0.5, type=float, help='maximum bandwidth for clipping')
        parser.add_argument('--num_freqs', default=10, type=int, help='number of frequency for sin/cos')

        # testing arguments
        parser.add_argument('--wav_loc', type=str, help='where to save predicted audio data if test points are given')
        parser.add_argument('--metric_loc', type=str, help='where to save/load error metrics')
        parser.add_argument('--error_file', default='errors', type=str, help='error file name without file extension')
        parser.add_argument('--recalculate_errors', default=1, type=bool_flag, help='calculate error metrics from the full dataset, if inactive, only polls the model at the test points')
        parser.add_argument('--test_points', nargs='*', default=['0_21', '0_89', '0_199'], help='point pairs \'src_rcv\' where to save and plot predicted audio data, if empty, nothing is saved and full errors are calculated')
        parser.add_argument('--gt_has_phase', default=0, type=bool_flag)  # image2reverb does not use gt phase for their GT when computing T60 error, and instead use random phase. If we use GT waveform (instead of randomizing the phase, we get lower T60 error)

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.check_paths()
        self.opt.freq_bins = self.opt.n_fft // 2
        self.opt.max_len = {'test_1': 86}  # Calculated when generating the dataset, 45 with fft_size 512
        self.opt.subj_offset = int(self.opt.grid.split('x')[0]) * int(self.opt.grid.split('x')[1]) - 1  # Offset to convert between 'subjects' and points, see test_query.py before error metric calculation
        self.opt.components = int((int(self.opt.order) + 1) ** 2)
        torch.manual_seed(0)
        np.random.seed(0)
        args = vars(self.opt)
        print('Arguments for the current run:')
        for k, v in args.items():
            print('    %s: %s' % (str(k), str(v)))
        print()
        return self.opt
