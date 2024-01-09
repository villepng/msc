import h5py
import librosa
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import tqdm

from scipy.io import wavfile
from scipy.signal import fftconvolve

from model.modules import EmbeddingModuleLog
from model.networks import KernelResidualFCEmbeds
from options import Options

'''
    Code from https://github.com/aluo-x/Learning_Neural_Acoustic_Fields
'''


def embed_input(args, rcv_pos, src_pos, max_len, min_pos, max_pos, output_device):
    xyz_embedder = EmbeddingModuleLog(num_freqs=args.num_freqs, ch_dim=2, max_freq=7).to(output_device)
    time_embedder = EmbeddingModuleLog(num_freqs=args.num_freqs, ch_dim=2).to(output_device)
    freq_embedder = EmbeddingModuleLog(num_freqs=args.num_freqs, ch_dim=2).to(output_device)

    transformed_input = prepare_input(0, rcv_pos, src_pos, max_len, min_pos, max_pos)
    degree = torch.Tensor([transformed_input[0]]).to(output_device, non_blocking=True).long()
    position = transformed_input[1][None].to(output_device, non_blocking=True)
    non_norm_position = transformed_input[2].to(output_device, non_blocking=True)
    freqs = transformed_input[3][None].to(output_device, non_blocking=True).unsqueeze(2) * 2.0 * math.pi
    times = transformed_input[4][None].to(output_device, non_blocking=True).unsqueeze(2) * 2.0 * math.pi
    pixel_count = max_len * 256

    position_embed = xyz_embedder(position).expand(-1, pixel_count, -1)
    freq_embed = freq_embedder(freqs)
    time_embed = time_embedder(times)

    return torch.cat((position_embed, freq_embed, time_embed), dim=2), degree, non_norm_position


def load_gt_data(args):
    spec_obj = h5py.File(f'{args.spec_base}/test_1.h5', 'r')
    phase_obj = h5py.File(f'metadata/mono/phases/test_1.h5', 'r')  # todo: add argument maybe

    with open(f'{args.coor_base}/{args.apt}/points.txt', 'r') as f:
        lines = f.readlines()
    points = [x.replace('\n', "").split(' ') for x in lines]
    positions = dict()
    for row in points:
        readout = [float(xyz) for xyz in row[1:]]
        positions[row[0]] = [readout[0], readout[1]]

    train_test_split = load_pkl(f'{args.split_loc}/{args.apt}_complete.pkl')
    train_keys = train_test_split[0]
    test_keys = train_test_split[1]

    return spec_obj, phase_obj, positions, train_keys, test_keys


def load_pkl(path):
    with open(path, 'rb') as loaded_pkl_obj:
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

    return degree, total_position, total_non_norm_position, 2.0 * torch.from_numpy(
        selected_freq).float() / 255.0 - 1.0, 2.0 * torch.from_numpy(selected_time).float() / float(max_len - 1) - 1.0


def prepare_network(weight_path, args, output_device, min_pos, max_pos):
    weights = torch.load(weight_path, map_location='cuda:0')
    auditory_net = KernelResidualFCEmbeds(input_ch=126, intermediate_ch=args.features,
                                          grid_ch=args.grid_features, num_block=args.layers, grid_gap=args.grid_gap,
                                          grid_bandwidth=args.bandwith_init, bandwidth_min=args.min_bandwidth,
                                          bandwidth_max=args.max_bandwidth, float_amt=args.position_float,
                                          min_xy=min_pos, max_xy=max_pos).to(output_device)
    auditory_net.load_state_dict(weights['network'])
    auditory_net.to('cuda:0')

    return auditory_net


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
    try:
        restored = (np.exp(padded_input_stft) - 1e-3) * phase_val
        wave1 = librosa.istft(restored[0], hop_length=512 // 4)
        # wave2 = librosa.istft(restored[1], hop_length=512 // 4)  # mono
        return wave1  # , wave2
    except:  # todo: proper fix in main
        return None


def main():
    args = Options().parse()
    apt = args.apt
    max_len = args.max_len[apt]
    weight_path = 'out/00200_2.chkpt'
    min_max = load_pkl(f'{args.minmax_base}/{args.apt}_minmax.pkl')
    min_pos = np.array(min_max[0][0:2])  # todo: make into np.array when created originally
    max_pos = np.array(min_max[1][0:2])
    output_device = 0
    orientation = 0

    # Load mean and std data & gt data and prepare the network
    mean_std = load_pkl(f'{args.mean_std_base}/{apt}.pkl')
    mean = torch.from_numpy(mean_std[0]).float()[None]
    std = 3.0 * torch.from_numpy(mean_std[1]).float()[None]
    spec_obj, phase_obj, points, train_keys, test_keys = load_gt_data(args)
    network = prepare_network(weight_path, args, output_device, min_pos, max_pos)

    # Poll network at every training data point and calculate metrics against gt data
    metrics = {'mse': [], 'spec': [], 't60': [], 'drr': [], 'errors': 0}  # todo
    progress = tqdm.tqdm(train_keys[orientation])
    progress.set_description('Polling network at every train data point to calculate error metrics')
    i = 0  # for tmp sanity check
    for key in progress:
        full_key = f'{orientation}_{key}'
        src, rcv = key.split('_')
        src_pos, rcv_pos = points[src], points[rcv]
        spec_data, phase_data = torch.from_numpy(spec_obj[full_key][:]).float(), torch.from_numpy(phase_obj[full_key][:]).float()
        spec_data, phase_data = spec_data[:, :, :max_len], phase_data[:, :, :max_len]
        try:
            spec_data, phase_data = (spec_data.reshape(1, 1, 256, max_len).cpu()).numpy(), (phase_data.reshape(1, 1, 256, max_len).cpu()).numpy()
        except:  # tmp fix
            pass

        # Poll the network
        net_input, degree, non_norm_position = embed_input(args, rcv_pos, src_pos, max_len, min_pos, max_pos, output_device)
        network.eval()
        with torch.no_grad():
            output = network(net_input, degree, non_norm_position.squeeze(1)).squeeze(3).transpose(1, 2)
        # output = (output.reshape(1, 1, 256, max_len).cpu() * std[None] + mean[None]).numpy()
        output = (output.reshape(1, 1, 256, max_len).cpu() * std[None] + mean[None]).numpy()

        # Convert into time domain to calculate metrics
        predicted_wave = to_wave_if(output[0], phase_data[0])  # using original phases
        gt_wave = to_wave_if(spec_data[0], phase_data[0])  # Could also load original RIR, but shouldn't matter

        if predicted_wave is not None and gt_wave is not None:
            metrics['mse'].append(np.square(np.subtract(predicted_wave, gt_wave)).mean())
        else:
            metrics['errors'] += 1

        # todo: plotting function?
        if i < 10 or key == '0_199':
            fig, axarr = plt.subplots(1, 2)
            fig.suptitle('Predicted log impulse response', fontsize=16)
            axarr[0].imshow(output[0, 0], cmap='inferno', vmin=np.min(output) * 1.1, vmax=np.max(output) * 0.9)
            axarr[0].set_title('Predicted')
            axarr[0].axis('off')
            axarr[1].imshow(spec_data[0, 0], cmap='inferno', vmin=np.min(spec_data) * 1.1, vmax=np.max(spec_data) * 0.9)
            axarr[1].set_title('Ground-truth')
            axarr[1].axis('off')
            plt.show()

            sr = 16000
            fig, axarr = plt.subplots(3, 1)
            fig.suptitle(f'Predicted impulse response {key}', fontsize=16)
            axarr[0].plot(np.arange(len(predicted_wave)) / sr, predicted_wave)
            axarr[0].set_xlim([0, 0.2])
            axarr[0].set_ylim([None, max(gt_wave) * 1.1])
            axarr[0].set_title('Predicted')
            axarr[1].plot(np.arange(len(gt_wave)) / sr, gt_wave)
            axarr[1].set_xlim([0, 0.2])
            axarr[1].set_ylim([None, max(gt_wave) * 1.1])
            axarr[1].set_title('Ground-truth')
            axarr[2].plot(np.arange(len(np.subtract(predicted_wave, gt_wave))) / sr, np.subtract(predicted_wave, gt_wave))
            axarr[2].set_ylim([None, max(gt_wave) * 1.1])
            axarr[2].set_xlim([0, 0.2])
            axarr[2].set_title('Error')
            plt.show()
        i += 1

    spec_obj.close()
    phase_obj.close()

    print(f'Avg. MSE: {np.average(metrics["mse"])}, errors: {metrics["errors"]}')


if __name__ == '__main__':
    main()
    """ Plotting
    fig, axarr = plt.subplots(1, 2)
    fig.suptitle('Predicted log impulse response', fontsize=16)
    axarr[0].imshow(output[0, 0], cmap='inferno', vmin=np.min(output) * 1.1, vmax=np.max(output) * 0.9)
    axarr[0].set_title('Predicted')
    axarr[0].axis('off')
    axarr[1].imshow(spec_data[0, 0], cmap='inferno', vmin=np.min(spec_data) * 1.1, vmax=np.max(spec_data) * 0.9)
    axarr[1].set_title('Ground-truth')
    axarr[1].axis('off')
    plt.show()

    # To wave test
    sr = 16000  # Currently 16k for all data, training data was originally resampled
    originals = load_pkl('/worktmp/melandev/data/generated/rirs/order_0/room_10.0x6.0x2.5/grid_20x10/rt60_0.2/rirs.pickle')
    original = originals['0-199']
    predicted_wave = to_wave_if(output[0], phase_data[0])  # using original phases
    gt_wave = to_wave_if(spec_data[0], phase_data[0])
    fig, axarr = plt.subplots(3, 1)
    fig.suptitle('Predicted impulse response', fontsize=16)
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
    fs, mono = wavfile.read('/worktmp/melandev/data/generated/rir_ambisonics_order_0_20x10/trainset/subject199/mono.wav')
    wave_rir_out = fftconvolve(mono, predicted_wave)
    wavfile.write(f'./out/predicted_wave.wav', fs, wave_rir_out.astype(np.int16))
    
    """
