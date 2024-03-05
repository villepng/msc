import h5py
import numpy as np
import os
import pickle
import random
import torch

from scipy.io import wavfile

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def join(*paths):
    return os.path.join(*paths)


def listdir(path):
    return [os.path.join(path, x) for x in os.listdir(path)]


class Soundsamples(torch.utils.data.Dataset):
    def __init__(self, arg_stuff):
        print('Starting dataset caching')
        coor_base = arg_stuff.coor_base
        spec_base = arg_stuff.spec_base
        mean_std_base = arg_stuff.mean_std_base
        minmax_base = arg_stuff.minmax_base
        early_base = arg_stuff.early_base
        room_name = arg_stuff.apt
        num_samples = arg_stuff.pixel_count

        coor_path = os.path.join(coor_base, room_name, 'points.txt')
        max_len = arg_stuff.max_len
        self.max_len = max_len[room_name]
        full_path = os.path.join(spec_base, room_name+'.h5')

        self.sound_data = []
        self.sound_data = h5py.File(full_path, 'r')
        self.sound_keys = list(self.sound_data.keys())
        self.sound_data.close()
        self.sound_data = None
        self.full_path = full_path
        self.phase_path = f'{arg_stuff.phase_base}/{room_name}.h5'
        self.components = int((int(arg_stuff.order) + 1) ** 2)

        self.sound_files = {'0': [], '90': [], '180': [], '270': []}
        self.sound_files_test = {'0': [], '90': [], '180': [], '270': []}

        train_test_split_path = os.path.join(arg_stuff.split_loc, arg_stuff.apt + '_complete.pkl')  # change this to f and also fix in options.py
        with open(train_test_split_path, 'rb') as train_test_file_obj:
            train_test_split = pickle.load(train_test_file_obj)

        self.sound_files = train_test_split[0]
        self.sound_files_test = train_test_split[1]

        self.early_data = {}
        self.early_data_test = {}
        dirs = os.listdir(early_base)
        for file in dirs:
            _, wav = wavfile.read(f'{early_base}/{file}')
            file = file.replace('.wav', '').replace('-', '_')
            if file in train_test_split[0][0]:  # suboptimal
                self.early_data.update({file: wav})
            else:
                self.early_data_test.update({file: wav})

        with open(os.path.join(mean_std_base, room_name+'.pkl'), 'rb') as mean_std_ff:
            mean_std = pickle.load(mean_std_ff)
        self.mean = torch.from_numpy(mean_std[0]).float()
        self.std = 3.0 * torch.from_numpy(mean_std[1]).float()
        self.std_if = 3.0 * torch.from_numpy(mean_std[2]).float()

        with open(coor_path, 'r') as f:
            lines = f.readlines()
        coords = [x.replace('\n', '').split(' ') for x in lines]  # use \t for the original way
        self.positions = dict()
        for row in coords:
            readout = [float(xyz) for xyz in row[1:]]
            self.positions[row[0]] = [readout[0], readout[1]]  # originally - for [1]??

        with open(os.path.join(minmax_base, room_name+'_minmax.pkl'), 'rb') as min_max_loader:
            min_maxes = pickle.load(min_max_loader)
            self.min_pos = min_maxes[0][0:2]
            self.max_pos = min_maxes[1][0:2]  # Different dimensions are the floor plane, problematic?
            self.min_pos = np.array(self.min_pos)  # tmp fix, to remove
            self.max_pos = np.array(self.max_pos)
            # self.min_pos = min_maxes[0][[0, 2]]
            # self.max_pos = min_maxes[1][[0, 2]]
            # This is since dimension 0 and 2 are floor plane

        # values = np.array(list(self.positions.values()))
        self.num_samples = num_samples
        self.pos_reg_amt = arg_stuff.reg_eps
        print('Finished dataset caching')

    def __len__(self):
        # return number of samples for a SINGLE orientation
        return len(list(self.sound_files.values())[0])

    def __getitem__(self, idx):
        loaded = False
        orientations = ['0']  # , '90', '180', '270']
        while not loaded:
            orientation_idx = 0  # random.randint(0, 3)
            orientation = orientations[orientation_idx]

            if self.sound_data is None:
                self.sound_data = h5py.File(self.full_path, 'r')
                self.phase_data = h5py.File(self.phase_path, 'r')

            pos_id = self.sound_files[int(orientation)][idx]
            position = (pos_id.split('.')[0]).split('_')
            query_str = orientation + '_' + pos_id

            spec_data = torch.from_numpy(self.sound_data[query_str][:]).float()
            phase_data = torch.from_numpy(self.phase_data[query_str][:]).float()
            spec_data, phase_data = spec_data[:, :, :self.max_len], phase_data[:, :, :self.max_len]
            early_data = torch.from_numpy(self.early_data[pos_id]).float()

            if random.random() < 0.1:
                # np.log(1e-3) = -6.90775527898213
                spec_data = torch.nn.functional.pad(spec_data, pad=[0, self.max_len-spec_data.shape[2], 0, 0, 0, 0], value=-6.90775527898213)
                phase_data = torch.nn.functional.pad(phase_data, pad=[0, self.max_len - phase_data.shape[2], 0, 0, 0, 0], value=-6.90775527898213)  # ?

            actual_spec_len = spec_data.shape[2]
            spec_data = (spec_data - self.mean[:, :, :actual_spec_len])/self.std[:, :, :actual_spec_len]
            # phase_data = phase_data / self.std_if
            # 2, freq, time
            sound_size = spec_data.shape
            selected_time = np.random.randint(0, sound_size[2], self.num_samples)
            # selected_time_ph = np.random.randint(0, 13, self.num_samples)  # only select phase from the beginning, corresponding to ~0-50ms (.5×13×128÷16000)
            selected_time_early = np.random.randint(0, 800, self.num_samples)
            selected_freq = np.random.randint(0, sound_size[1], self.num_samples)
            degree = orientation_idx

            non_norm_start = (np.array(self.positions[position[0]])[:2] + np.random.normal(0, 1, 2)*self.pos_reg_amt)
            non_norm_end = (np.array(self.positions[position[1]])[:2] + np.random.normal(0, 1, 2)*self.pos_reg_amt)
            start_position = (torch.from_numpy((non_norm_start - self.min_pos)/(self.max_pos-self.min_pos))[None] - 0.5) * 2.0
            start_position = torch.clamp(start_position, min=-1.0, max=1.0)
            end_position = (torch.from_numpy((non_norm_end - self.min_pos)/(self.max_pos-self.min_pos))[None] - 0.5) * 2.0
            end_position = torch.clamp(end_position, min=-1.0, max=1.0)

            total_position = torch.cat((start_position, end_position), dim=1).float()
            total_non_norm_position = torch.cat((torch.from_numpy(non_norm_start)[None], torch.from_numpy(non_norm_end)[None]), dim=1).float()

            selected_total = spec_data[:, selected_freq, selected_time]
            # selected_total_phase = phase_data[:, selected_freq, selected_time]
            loaded = True

        return (selected_total, early_data, degree, total_position, total_non_norm_position,
                2.0*torch.from_numpy(selected_freq).float()/255.0 - 1.0,
                2.0*torch.from_numpy(selected_time).float()/float(self.max_len - 1) - 1.0,
                2.0*torch.from_numpy(selected_time_early).float()/799 - 1.0)

    def get_item_teaser(self, orientation_idx, reciever_pos, source_pos):
        selected_time = np.arange(0, self.max_len)
        selected_freq = np.arange(0, 256)
        selected_time, selected_freq = np.meshgrid(selected_time, selected_freq)
        selected_time = selected_time.reshape(-1)
        selected_freq = selected_freq.reshape(-1)

        degree = orientation_idx

        non_norm_start = np.array(reciever_pos)
        non_norm_end = np.array(source_pos)
        total_non_norm_position = torch.cat((torch.from_numpy(non_norm_start)[None], torch.from_numpy(non_norm_end)[None]), dim=1).float()

        start_position = (torch.from_numpy((non_norm_start - self.min_pos) / (self.max_pos - self.min_pos))[None] - 0.5) * 2.0
        start_position = torch.clamp(start_position, min=-1.0, max=1.0)
        end_position = (torch.from_numpy((non_norm_end - self.min_pos) / (self.max_pos - self.min_pos))[None] - 0.5) * 2.0
        end_position = torch.clamp(end_position, min=-1.0, max=1.0)
        total_position = torch.cat((start_position, end_position), dim=1).float()

        return degree, total_position, total_non_norm_position, 2.0*torch.from_numpy(selected_freq).float()/255.0 - 1.0, 2.0*torch.from_numpy(selected_time).float()/float(self.max_len-1)-1.0

    def get_item_test(self, orientation_idx, idx):
        orientations = ['0']  # , '90', '180', '270']
        orientation = orientations[orientation_idx]
        selected_files = self.sound_files_test
        if self.sound_data is None:
            self.sound_data = h5py.File(self.full_path, 'r')
        pos_id = selected_files[orientation][idx]
        query_str = orientation + '_' + pos_id
        spec_data = torch.from_numpy(self.sound_data[query_str][:]).float()

        position = (pos_id.split('.')[0]).split('_')

        spec_data = spec_data[:, :, :self.max_len]
        actual_spec_len = spec_data.shape[2]

        spec_data = (spec_data - self.mean[:, :, :actual_spec_len])/self.std[:, :, :actual_spec_len]
        # 2, freq, time
        sound_size = spec_data.shape
        self.sound_size = sound_size
        self.sound_name = position
        selected_time = np.arange(0, sound_size[2])
        selected_freq = np.arange(0, sound_size[1])
        selected_time, selected_freq = np.meshgrid(selected_time, selected_freq)
        selected_time = selected_time.reshape(-1)
        selected_freq = selected_freq.reshape(-1)

        degree = orientation_idx

        non_norm_start = np.array(self.positions[position[0]])[:2]
        non_norm_end = np.array(self.positions[position[1]])[:2]
        start_position = (torch.from_numpy((non_norm_start - self.min_pos) / (self.max_pos - self.min_pos))[None] - 0.5) * 2.0
        start_position = torch.clamp(start_position, min=-1.0, max=1.0)
        end_position = (torch.from_numpy((non_norm_end - self.min_pos) / (self.max_pos - self.min_pos))[None] - 0.5) * 2.0
        end_position = torch.clamp(end_position, min=-1.0, max=1.0)
        total_position = torch.cat((start_position, end_position), dim=1).float()
        total_non_norm_position = torch.cat((torch.from_numpy(non_norm_start)[None], torch.from_numpy(non_norm_end)[None]), dim=1).float()

        selected_total = spec_data[:, selected_freq, selected_time]
        return selected_total, degree, total_position, total_non_norm_position, 2.0*torch.from_numpy(selected_freq).float()/255.0 - 1.0, 2.0*torch.from_numpy(selected_time).float()/float(self.max_len-1)-1.0
