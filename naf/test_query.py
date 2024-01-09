import h5py
import librosa
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch

from scipy.io import wavfile
from scipy.signal import fftconvolve
from torch import nn


"""
    Code from https://github.com/aluo-x/Learning_Neural_Acoustic_Fields
"""


class embedding_module_log(nn.Module):
    def __init__(self, funcs=[torch.sin, torch.cos], num_freqs=20, max_freq=10, ch_dim=1, include_in=True):
        super().__init__()
        self.functions = funcs
        self.num_functions = list(range(len(funcs)))
        self.freqs = torch.nn.Parameter(2.0**torch.from_numpy(np.linspace(start=0.0,stop=max_freq, num=num_freqs).astype(np.single)), requires_grad=False)
        self.ch_dim = ch_dim
        self.funcs = funcs
        self.include_in = include_in

    def forward(self, x_input):
        if self.include_in:
            out_list = [x_input]
        else:
            out_list = []
        for func in self.funcs:
            for freq in self.freqs:
                out_list.append(func(x_input*freq))
        return torch.cat(out_list, dim=self.ch_dim)


class basic_project2(nn.Module):
    def __init__(self, input_ch, output_ch):
        super(basic_project2, self).__init__()
        self.proj = nn.Linear(input_ch, output_ch, bias=True)
    def forward(self, x):
        return self.proj(x)


class kernel_linear_act(nn.Module):
    def __init__(self, input_ch, output_ch):
        super(kernel_linear_act, self).__init__()
        self.block = nn.Sequential(nn.LeakyReLU(negative_slope=0.1), basic_project2(input_ch, output_ch))
    def forward(self, input_x):
        return self.block(input_x)


class kernel_residual_fc_embeds(nn.Module):
    def __init__(self, input_ch, intermediate_ch=512, grid_ch = 64, num_block=8, output_ch=1, grid_gap=0.25, grid_bandwidth=0.25, bandwidth_min=0.1, bandwidth_max=0.5, float_amt=0.1, min_xy=None, max_xy=None, probe=False):
        super(kernel_residual_fc_embeds, self).__init__()
        # input_ch (int): number of ch going into the network
        # intermediate_ch (int): number of intermediate neurons
        # min_xy, max_xy are the bounding box of the room in real (not normalized) coordinates
        # probe = True returns the features of the last layer

        for k in range(num_block - 1):
            self.register_parameter("left_right_{}".format(k),nn.Parameter(torch.randn(1, 1, 1, intermediate_ch)/math.sqrt(intermediate_ch),requires_grad=True))  # changed for mono

        for k in range(4):
            self.register_parameter("rot_{}".format(k), nn.Parameter(torch.randn(num_block - 1, 1, 1, intermediate_ch)/math.sqrt(intermediate_ch), requires_grad=True))

        self.proj = basic_project2(input_ch + int(2*grid_ch), intermediate_ch)
        self.residual_1 = nn.Sequential(basic_project2(input_ch + 128, intermediate_ch), nn.LeakyReLU(negative_slope=0.1), basic_project2(intermediate_ch, intermediate_ch))
        self.layers = torch.nn.ModuleList()
        for k in range(num_block - 2):
            self.layers.append(kernel_linear_act(intermediate_ch, intermediate_ch))

        self.out_layer = nn.Linear(intermediate_ch, output_ch)
        self.blocks = len(self.layers)
        self.probe = probe

        # Make the grid
        grid_coors_x = np.arange(min_xy[0], max_xy[0], grid_gap)
        grid_coors_y = np.arange(min_xy[1], max_xy[1], grid_gap)
        grid_coors_x, grid_coors_y = np.meshgrid(grid_coors_x, grid_coors_y)
        grid_coors_x = grid_coors_x.flatten()
        grid_coors_y = grid_coors_y.flatten()
        xy_train = np.array([grid_coors_x, grid_coors_y]).T
        self.bandwidth_min = bandwidth_min
        self.bandwidth_max = bandwidth_max
        self.float_amt = float_amt
        self.bandwidths = nn.Parameter(torch.zeros(len(grid_coors_x))+grid_bandwidth, requires_grad=True)
        self.register_buffer("grid_coors_xy",torch.from_numpy(xy_train).float(), persistent=True)
        self.xy_offset = nn.Parameter(torch.zeros_like(self.grid_coors_xy), requires_grad=True)
        self.grid_0 = nn.Parameter(torch.randn(len(grid_coors_x),grid_ch, device="cpu").float() / np.sqrt(float(grid_ch)), requires_grad=True)

    def forward(self, input_stuff, rot_idx, sound_loc=None):
        SAMPLES = input_stuff.shape[1]
        sound_loc_v0 = sound_loc[..., :2]
        sound_loc_v1 = sound_loc[..., 2:]

        # Prevent numerical issues
        self.bandwidths.data = torch.clamp(self.bandwidths.data, self.bandwidth_min, self.bandwidth_max)

        grid_coors_baseline = self.grid_coors_xy + torch.tanh(self.xy_offset) * self.float_amt
        grid_feat_v0 = fit_predict_torch(grid_coors_baseline, self.grid_0, sound_loc_v0, self.bandwidths)
        grid_feat_v1 = fit_predict_torch(grid_coors_baseline, self.grid_0, sound_loc_v1, self.bandwidths)
        total_grid = torch.cat((grid_feat_v0, grid_feat_v1), dim=-1).unsqueeze(1).expand(-1, SAMPLES, -1)

        my_input = torch.cat((total_grid, input_stuff), dim=-1)
        rot_latent = torch.stack([getattr(self, "rot_{}".format(rot_idx_single)) for rot_idx_single in rot_idx], dim=0)
        out = self.proj(my_input).unsqueeze(2).repeat(1, 1, 1, 1) + getattr(self, "left_right_0") + rot_latent[:, 0]  # changed for mono
        for k in range(len(self.layers)):
            out = self.layers[k](out) + getattr(self, "left_right_{}".format(k + 1)) + rot_latent[:, k + 1]
            if k == (self.blocks // 2 - 1):
                out = out + self.residual_1(my_input).unsqueeze(2).repeat(1, 1, 1, 1)  # changed for mono
        if self.probe:
            return out
        return self.out_layer(out)


def distance(x1, x2):
    # by jacobrgardner
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
    return res


def fit_predict_torch(input_pos:torch.Tensor, input_target:torch.Tensor, predict_pos:torch.Tensor, bandwidth:torch.Tensor) -> torch.Tensor:
    dist_vector = -distance(predict_pos, input_pos)
    gauss_dist = torch.exp(dist_vector/(2.0 * torch.square(bandwidth.unsqueeze(0))))
    magnitude = torch.sum(gauss_dist, dim=1, keepdim=True)
    out = torch.mm(gauss_dist, input_target)/magnitude
    return out


def load_pkl(path):
    with open(path, "rb") as loaded_pkl_obj:
        loaded_pkl = pickle.load(loaded_pkl_obj)
    return loaded_pkl


def prepare_input(orientation_idx, reciever_pos, source_pos, max_len, min_bbox_pos, max_bbox_pos):
    selected_time = np.arange(0, max_len)
    selected_freq = np.arange(0, 256)
    selected_time, selected_freq = np.meshgrid(selected_time, selected_freq)
    selected_time = selected_time.reshape(-1)
    selected_freq = selected_freq.reshape(-1)

    degree = orientation_idx

    non_norm_start = np.array(reciever_pos)
    non_norm_end = np.array(source_pos)
    total_non_norm_position = torch.cat((torch.from_numpy(non_norm_start)[None], torch.from_numpy(non_norm_end)[None]), dim=1).float()

    start_position = (torch.from_numpy((non_norm_start - min_bbox_pos) / (max_bbox_pos - min_bbox_pos))[None] - 0.5) * 2.0
    start_position = torch.clamp(start_position, min=-1.0, max=1.0)
    end_position = (torch.from_numpy((non_norm_end - min_bbox_pos) / (max_bbox_pos - min_bbox_pos))[None] - 0.5) * 2.0
    end_position = torch.clamp(end_position, min=-1.0, max=1.0)
    total_position = torch.cat((start_position, end_position), dim=1).float()

    return degree, total_position, total_non_norm_position, 2.0*torch.from_numpy(selected_freq).float()/255.0 - 1.0, 2.0*torch.from_numpy(selected_time).float()/float(max_len-1)-1.0


def to_wave(input_spec, mean_val=None, std_val=None, gl=False, orig_phase=None):
    if not mean_val is None:
        renorm_input = input_spec * std_val
        renorm_input = renorm_input + mean_val
    else:
        renorm_input = input_spec + 0.0
    renorm_input = np.exp(renorm_input) - 1e-3
    renorm_input = np.clip(renorm_input, 0.0, 100000.0)
    if orig_phase is None:
        if gl is False:
            # Random phase reconstruction per image2reverb
            np.random.seed(1234)
            rp = np.random.uniform(-np.pi, np.pi, renorm_input.shape)
            f = renorm_input * (np.cos(rp) + (1.j * np.sin(rp)))
            out_wave = librosa.istft(f, hop_length=128)
        else:
            out_wave = librosa.griffinlim(renorm_input, hop_length=128, n_iter=40, momentum=0.5, random_state=64)
    else:
        f = renorm_input * (np.cos(orig_phase) + (1.j * np.sin(orig_phase)))
        out_wave = librosa.istft(f, win_length=400, hop_length=200)
    return out_wave


def to_wave_if(input_stft, input_if):
    # 2 chanel input of shape [2,freq,time]
    # First input is logged mag
    # Second input is if divided by np.pi
    padded_input_stft = np.concatenate((input_stft, input_stft[:, -1:]), axis=1)
    padded_input_if = np.concatenate((input_if, input_if[:, -1:] * 0.0), axis=1)
    unwrapped = np.cumsum(padded_input_if, axis=-1) * np.pi
    phase_val = np.cos(unwrapped) + 1j * np.sin(unwrapped)
    restored = (np.exp(padded_input_stft) - 1e-3) * phase_val
    wave1 = librosa.istft(restored[0], hop_length=512 // 4)
    # wave2 = librosa.istft(restored[1], hop_length=512 // 4)  # mono
    return wave1  # , wave2


def main():
    weights = torch.load('out/00200_2.chkpt', map_location='cuda:0')  # chkpt file
    min_pos = np.array([0, 0])
    max_pos = np.array([10, 6])
    apt = "test_1"
    max_lengths = {"test_1": 28}  # 35
    output_device = 0
    num_freqs = 10

    xyz_embedder = embedding_module_log(num_freqs=num_freqs, ch_dim=2, max_freq=7).to(output_device)
    time_embedder = embedding_module_log(num_freqs=num_freqs, ch_dim=2).to(output_device)
    freq_embedder = embedding_module_log(num_freqs=num_freqs, ch_dim=2).to(output_device)
    auditory_net = kernel_residual_fc_embeds(input_ch=126, intermediate_ch=512, grid_ch=64, num_block=8, grid_gap=0.25,
                                             grid_bandwidth=0.25, bandwidth_min=0.1, bandwidth_max=0.5, float_amt=0.1,
                                             min_xy=min_pos, max_xy=max_pos)
    loaded = auditory_net.load_state_dict(weights["network"])
    loaded = auditory_net.to("cuda:0")

    emitter_position = np.array([1.0, 1.0])  # set manually for now
    listener_position = np.array([9.0, 5.0])
    orientation = 0

    transformed_input = prepare_input(0, listener_position, emitter_position, max_lengths[apt], min_pos, max_pos)

    # Now we apply sinusoidal embeddings:
    degree = torch.Tensor([transformed_input[0]]).to(output_device, non_blocking=True).long()
    position = transformed_input[1][None].to(output_device, non_blocking=True)
    non_norm_position = transformed_input[2].to(output_device, non_blocking=True)
    freqs = transformed_input[3][None].to(output_device, non_blocking=True).unsqueeze(2) * 2.0 * math.pi
    times = transformed_input[4][None].to(output_device, non_blocking=True).unsqueeze(2) * 2.0 * math.pi
    PIXEL_COUNT = max_lengths[apt] * 256
    position_embed = xyz_embedder(position).expand(-1, PIXEL_COUNT, -1)
    freq_embed = freq_embedder(freqs)
    time_embed = time_embedder(times)
    total_in = torch.cat((position_embed, freq_embed, time_embed), dim=2)

    output_container = []
    auditory_net.eval()
    with torch.no_grad():
        output = auditory_net(total_in, degree, non_norm_position.squeeze(1)).squeeze(3).transpose(1, 2)

    data_path = "../../code/Learning_Neural_Acoustic_Fields-master/metadata_mono"
    # Load mean and std data
    mean_std = load_pkl(f"{data_path}/mean_std/test_1.pkl")
    mean = torch.from_numpy(mean_std[0]).float()[None]
    std = 3.0 * torch.from_numpy(mean_std[1]).float()[None]
    output = (output.reshape(1, 1, 256, max_lengths[apt]).cpu() * std[None] + mean[None]).numpy()
    print("Completed inference")

    # Load gt data, todo: use point data to calculate stuff for test points and poll the model at those points
    spec_obj = h5py.File(f"{data_path}/magnitudes/test_1.h5", 'r')
    spec_data = torch.from_numpy(spec_obj["0_0_199"][:]).float()
    spec_obj.close()
    spec_data = spec_data[:, :, : max_lengths[apt]]
    spec_data = (spec_data.reshape(1, 1, 256, max_lengths[apt]).cpu()).numpy()

    phase_obj = h5py.File(f"../../data/naf/order_0/phases/test_1.h5", 'r')
    phase_data = torch.from_numpy(phase_obj["0_0_199"][:]).float()
    phase_obj.close()
    phase_data = phase_data[:, :, : max_lengths[apt]]
    phase_data = (phase_data.reshape(1, 1, 256, max_lengths[apt]).cpu()).numpy()

    fig, axarr = plt.subplots(1, 2)
    fig.suptitle("Predicted log impulse response", fontsize=16)
    axarr[0].imshow(output[0, 0], cmap="inferno", vmin=np.min(output) * 1.1, vmax=np.max(output) * 0.9)
    axarr[0].set_title('Predicted')
    axarr[0].axis("off")
    axarr[1].imshow(spec_data[0, 0], cmap="inferno", vmin=np.min(spec_data) * 1.1, vmax=np.max(spec_data) * 0.9)
    axarr[1].set_title('Ground-truth')
    axarr[1].axis("off")
    plt.show()

    # To wave test
    sr = 16000  # Currently 16k for all data, training data was originally resampled
    originals = load_pkl("/worktmp/melandev/data/generated/rirs/order_0/room_10.0x6.0x2.5/grid_20x10/rt60_0.2/rirs.pickle")
    original = originals["0-199"]
    predicted_wave = to_wave_if(output[0], phase_data[0])  # using original phases
    gt_wave = to_wave_if(spec_data[0], phase_data[0])
    fig, axarr = plt.subplots(3, 1)
    fig.suptitle("Predicted impulse response", fontsize=16)
    axarr[0].plot(np.arange(len(predicted_wave)) / sr, predicted_wave)
    axarr[0].set_xlim([0, 0.2])
    axarr[0].set_title('Predicted')
    axarr[1].plot(np.arange(len(gt_wave)) / sr, gt_wave)
    axarr[1].set_xlim([0, 0.2])
    axarr[1].set_title('Ground-truth')
    axarr[2].plot(np.arange(len(original)) / sr, original)
    axarr[2].set_xlim([0, 0.2])
    axarr[2].set_title('Original')
    plt.show()

    # Audio test
    fs, mono = wavfile.read("/worktmp/melandev/data/generated/rir_ambisonics_order_0_20x10/trainset/subject199/mono.wav")
    wave_rir_out = fftconvolve(mono, predicted_wave)
    wavfile.write(f'./out/predicted_wave.wav', fs, wave_rir_out.astype(np.int16))


if __name__ == '__main__':
    main()
