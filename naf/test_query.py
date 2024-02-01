import h5py
import librosa
import math
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pickle
import torch
import tqdm

import naf.metrics as metrics

from pyroomacoustics.experimental.rt60 import measure_rt60
from scipy.io import wavfile
from scipy.signal import fftconvolve

from naf.model.modules import EmbeddingModuleLog
from naf.model.networks import KernelResidualFCEmbeds
from naf.options import Options


METRICS_BAND = ['mse', 'rt60', 'drr', 'c50', 'errors']
METRICS_CHANNEL = ['spec_err', 'mse_wav']


def embed_input(args, rcv_pos, src_pos, max_len, min_pos, max_pos, output_device):
    xyz_embedder = EmbeddingModuleLog(num_freqs=args.num_freqs, ch_dim=2, max_freq=7).to(output_device)
    time_embedder = EmbeddingModuleLog(num_freqs=args.num_freqs, ch_dim=2).to(output_device)
    freq_embedder = EmbeddingModuleLog(num_freqs=args.num_freqs, ch_dim=2).to(output_device)

    transformed_input = prepare_input(0, rcv_pos, src_pos, max_len, min_pos, max_pos, args.freq_bins)
    degree = torch.Tensor([transformed_input[0]]).to(output_device, non_blocking=True).long()
    position = transformed_input[1][None].to(output_device, non_blocking=True)
    non_norm_position = transformed_input[2].to(output_device, non_blocking=True)
    freqs = transformed_input[3][None].to(output_device, non_blocking=True).unsqueeze(2) * 2.0 * math.pi
    times = transformed_input[4][None].to(output_device, non_blocking=True).unsqueeze(2) * 2.0 * math.pi
    times_ph = transformed_input[5][None].to(output_device, non_blocking=True).unsqueeze(2) * 2.0 * math.pi
    pixel_count = max_len * args.freq_bins

    position_embed = xyz_embedder(position).expand(-1, pixel_count, -1)
    freq_embed = freq_embedder(freqs)
    time_embed = time_embedder(times)
    time_embed_ph = time_embedder(times_ph)

    return torch.cat((position_embed, freq_embed, time_embed), dim=2), degree, non_norm_position


def get_error_metric_dict(components, band_centerfreqs):
    global METRICS_BAND, METRICS_CHANNEL
    error_metrics = {'train': {}, 'test': {}}
    for train_test in ['train', 'test']:
        for channel in range(components):
            error_metrics[train_test].update({channel: {}})
            for total in METRICS_CHANNEL:
                error_metrics[train_test][channel].update({total: []})
            for band in band_centerfreqs:
                error_metrics[train_test][channel].update({band: {}})
                for metric in METRICS_BAND:
                    error_metrics[train_test][channel][band].update({metric: []})
    return error_metrics


def load_gt_data(args):
    spec_obj = h5py.File(f'{args.spec_base}/test_1.h5', 'r')
    phase_obj = h5py.File(f'{args.phase_base}/test_1.h5', 'r')
    with open(f'metadata/ambisonics_{args.order}_{args.grid}/normalization/{args.apt}_max_val.txt', 'r') as f:
        max_val = f.readlines()

    with open(f'{args.coor_base}/{args.apt}/points.txt', 'r') as f:
        lines = f.readlines()
    points = [x.replace('\n', '').split(' ') for x in lines]
    positions = dict()
    for row in points:
        readout = [float(xyz) for xyz in row[1:]]
        positions[row[0]] = [readout[0], readout[1]]

    train_test_split = load_pkl(f'{args.split_loc}/{args.apt}_complete.pkl')
    train_keys = train_test_split[0]
    test_keys = train_test_split[1]

    return spec_obj, phase_obj, positions, train_keys, test_keys, float(max_val[0])


def load_pkl(path):
    with open(path, 'rb') as loaded_pkl_obj:
        loaded_pkl = pickle.load(loaded_pkl_obj)
    return loaded_pkl


def plot_stft(pred, gt, points):
    fig, axarr = plt.subplots(1, 3)
    fig.suptitle(f'Predicted log impulse response {points}', fontsize=16)
    axarr[0].imshow(pred[0, 0], cmap='inferno', vmin=np.min(gt) * 1.1, vmax=np.max(gt) * 0.9)
    axarr[0].set_title('Predicted')
    # axarr[0].axis('off')
    axarr[1].imshow(gt[0, 0], cmap='inferno', vmin=np.min(gt) * 1.1, vmax=np.max(gt) * 0.9)
    axarr[1].set_title('Ground-truth')
    # axarr[1].axis('off')
    axarr[2].imshow(gt[0, 0] - pred[0, 0], cmap='inferno', vmin=np.min(gt) * 1.1, vmax=np.max(gt) * 0.9)
    axarr[2].set_title('Error')
    # axarr[2].axis('off')
    plt.show()


def plot_wave(pred, gt, points, name='impulse response', sr=16000):
    fig, axarr = plt.subplots(3, 1)
    fig.suptitle(f'Predicted vs. GT {name} between {points}', fontsize=16)
    max_len = max(np.arange(len(gt)) / sr)
    axarr[0].plot(np.arange(len(pred)) / sr, pred)
    axarr[0].set_xlim([0, max_len])
    axarr[0].set_ylim([None, max(gt) * 1.1])
    axarr[0].set_title('Predicted')
    axarr[1].plot(np.arange(len(gt)) / sr, gt)
    axarr[1].set_xlim([0, max_len])
    axarr[1].set_ylim([None, max(gt) * 1.1])
    axarr[1].set_title('Ground-truth')
    axarr[2].plot(np.arange(len(np.subtract(pred[:len(gt)], gt))) / sr, np.subtract(pred[:len(gt)], gt))
    axarr[2].set_ylim([None, max(gt) * 1.1])
    axarr[2].set_xlim([0, max_len])
    axarr[2].set_title('Error')
    plt.show()


def prepare_input(orientation_idx, reciever_pos, source_pos, max_len, min_bbox_pos, max_bbox_pos, freq_bins):
    selected_time = np.arange(0, max_len)
    selected_freq = np.arange(0, freq_bins)
    selected_time, selected_freq = np.meshgrid(selected_time, selected_freq)
    selected_time = selected_time.reshape(-1)
    selected_freq = selected_freq.reshape(-1)

    selected_time_ph = np.arange(0, 13)
    selected_time_ph, _ = np.meshgrid(selected_time_ph, selected_freq)
    selected_time_ph = selected_time.reshape(-1)

    degree = orientation_idx

    non_norm_start = np.array(reciever_pos)
    non_norm_end = np.array(source_pos)
    total_non_norm_position = torch.cat((torch.from_numpy(non_norm_start)[None], torch.from_numpy(non_norm_end)[None]), dim=1).float()

    start_position = (torch.from_numpy((non_norm_start - min_bbox_pos) / (max_bbox_pos - min_bbox_pos))[None] - 0.5) * 2.0
    start_position = torch.clamp(start_position, min=-1.0, max=1.0)
    end_position = (torch.from_numpy((non_norm_end - min_bbox_pos) / (max_bbox_pos - min_bbox_pos))[None] - 0.5) * 2.0
    end_position = torch.clamp(end_position, min=-1.0, max=1.0)
    total_position = torch.cat((start_position, end_position), dim=1).float()

    return (degree, total_position, total_non_norm_position, 2.0 * torch.from_numpy(selected_freq).float() / 255.0 - 1.0,
            2.0 * torch.from_numpy(selected_time).float() / float(max_len - 1) - 1.0, 2.0 * torch.from_numpy(selected_time_ph).float() / float(12) - 1.0)


def prepare_network(weight_path, args, output_device, min_pos, max_pos):
    weights = torch.load(weight_path, map_location='cuda:0')
    auditory_net = KernelResidualFCEmbeds(input_ch=126, intermediate_ch=args.features, grid_ch=args.grid_features, num_block=args.layers,
                                          grid_gap=args.grid_gap, grid_bandwidth=args.bandwith_init, bandwidth_min=args.min_bandwidth,
                                          bandwidth_max=args.max_bandwidth, float_amt=args.position_float, min_xy=min_pos, max_xy=max_pos,
                                          components=args.components).to(output_device)
    auditory_net.load_state_dict(weights['network'])
    auditory_net.to('cuda:0')

    return auditory_net


def print_errors(error_metrics):  # train, channel, band, metric
    global METRICS_CHANNEL
    for train_test in ['train', 'test']:
        print(f'Errors for {train_test} points')
        for channel in error_metrics[train_test]:
            print(f'  channel {channel}')
            print(f'    avg. RIR spectral error: {np.average(error_metrics[train_test][channel]["spec_err"]):.6f}')
            # if len(error_metrics[train_test][channel]["mse_wav"]) > 0:  # add if necessary
            for data in error_metrics[train_test][channel]:
                if data not in METRICS_CHANNEL:
                    print(f'    band: {data}')
                    for metric in error_metrics[train_test][channel][data]:
                        if len(error_metrics[train_test][channel][data][metric]) > 0:
                            print(f'      avg. {metric.upper()}: {np.average(error_metrics[train_test][channel][data][metric]):.6f}')


def print_errors_old(error_metrics):
    for train_test in ['train', 'test']:
        print(f'{train_test} points'
              f'\n  avg. MSE for RIRs:   {np.average(error_metrics[train_test]["mse"]):.6f}'
              f'\n  avg. spectral error: {np.average(error_metrics[train_test]["spec_mse"]):.6f}'
              f'\n  avg. RT60 error:     {np.average(error_metrics[train_test]["rt60"]):.6f}'
              f'\n  avg. DRR error (dB): {np.average(error_metrics[train_test]["drr"]):.6f}'
              f'\n  avg. C50 error (dB): {np.average(error_metrics[train_test]["c50"]):.6f}')
        if len(error_metrics[train_test]["mse_wav"]) > 0:
            print(f'\n  avg. MSE for the reverberant audio waveforms: {np.average(error_metrics[train_test]["mse_wav"])}:.6f')
        if error_metrics[train_test]["errors"] != 0:
            print(f'  errors: {error_metrics[train_test]["errors"]}')  # currently not used at all


def test_model(args, test_points=None, write_errors=True):
    apt = args.apt
    max_len = args.max_len[apt]
    weight_path = f'{args.model_save_loc}/{apt}/0200.chkpt'
    min_max = load_pkl(f'{args.minmax_base}/{args.apt}_minmax.pkl')
    min_pos, max_pos = np.array(min_max[0][0:2]), np.array(min_max[1][0:2])
    output_device = 0
    orientation = 0

    # Load mean and std data & gt data and prepare the network
    mean_std = load_pkl(f'{args.mean_std_base}/{apt}.pkl')
    mean = torch.from_numpy(mean_std[0]).float()
    std = 3.0 * torch.from_numpy(mean_std[1]).float()
    # std_phase = 3.0 * torch.from_numpy(mean_std[2]).float()
    spec_obj, phase_obj, points, train_keys, test_keys, max_val = load_gt_data(args)
    if test_points is not None:
        train_keys[orientation] = []
        test_keys[orientation] = test_points
    network = prepare_network(weight_path, args, output_device, min_pos, max_pos)

    # Polling the network to calculate error metrics
    band_centerfreqs = np.array([125, 250, 500, 1000, 2000, 4000])
    bands = len(band_centerfreqs)
    error_metrics = get_error_metric_dict(args.components, band_centerfreqs)  # train, channel, band, metrics

    for train_test, keys in {'train': train_keys[orientation], 'test': test_keys[orientation]}.items():
        progress = tqdm.tqdm(keys)
        progress.set_description(f'Polling network to calculate error metrics at {train_test} data points')
        for i, key in enumerate(progress):
            full_key = f'{orientation}_{key}'
            src, rcv = key.split('_')
            src_pos, rcv_pos = points[src], points[rcv]
            spec_data0, phase_data0 = spec_obj[full_key][:], phase_obj[full_key][:]
            try:
                spec_data, phase_data = ((spec_data0.reshape(1, args.components, args.freq_bins, max_len)),
                                         (phase_data0.reshape(1, args.components, args.freq_bins, max_len)))
            except ValueError:  # pad with zeros if lengths are not equal to max_len
                len_spec, len_phase = spec_data0.shape[-1], phase_data0.shape[-1]
                spec_data0, phase_data0 = ((spec_data0.reshape(1, args.components, args.freq_bins, len_spec)),
                                           (phase_data0.reshape(1, args.components, args.freq_bins, len_phase)))
                spec_data, phase_data = (np.zeros((1, args.components, args.freq_bins, max_len), dtype=np.int32),
                                         np.zeros((1, args.components, args.freq_bins, max_len), dtype=np.int32))
                spec_data[:, :, :, :len_spec], phase_data[:, :, :, :len_phase] = spec_data0, phase_data0

            # Poll the network
            net_input, degree, non_norm_position = embed_input(args, rcv_pos, src_pos, max_len, min_pos, max_pos, output_device)
            network.eval()
            with torch.no_grad():
                output = network(net_input, degree, non_norm_position.squeeze(1)).squeeze(3).transpose(1, 2)
            # phase = output[:, :, :, 1]
            # output = output[:, :, :, 0]
            # phase = (output.reshape(1, args.components, args.freq_bins, max_len).cpu() * std_phase).numpy()
            output = (output.reshape(1, args.components, args.freq_bins, max_len).cpu() * std + mean).numpy()

            # Random phase reconstruction per image2reverb for later parts of the RIR
            # full_phase = np.zeros([1, args.components, args.freq_bins, max_len])
            # full_phase[:, :, :, :13] = phase[:, :, :, :13]
            # np.random.seed(1234)
            # rp = np.random.uniform(-np.pi, np.pi, [1, args.components, args.freq_bins, max_len - 13])
            # full_phase[:, :, :, 13:] = rp

            # Convert into time domain to calculate most metrics
            # predicted_rir = to_wave_if(output[0], phase_data[0], args.hop_len)  # using original phases
            predicted_rir = to_wave(output[0], args.hop_len)  # [channels, length], random phase
            # predicted_rir = to_wave_if(output[0], phase[0], args.hop_len)  # predicted phase
            # predicted_rir = to_wave_if(spec_data[0], phase[0], args.hop_len)  # predicted and random phase
            gt_rir = to_wave_if(spec_data[0], phase_data[0], args.hop_len)  # could also load original RIR, but shouldn't matter
            # plot_stft(phase, phase_data, key)
            # gt_rir = to_wave(spec_data[0])[0]  # test reconstructing GT with random phase
            # t = np.arange(len(gt_rir)) / 16000
            # plt.plot(t, 20*np.log10(abs(gt_rir)))
            # plt.show()

            # Convert from src and rcv points into 'subjects' in the original dataset format
            src, rcv = int(src), int(rcv)
            if rcv < src:
                subj = src * args.subj_offset + rcv + 1
            else:
                subj = src * args.subj_offset + rcv
            fs, mono = wavfile.read(f'{args.wav_base}/trainset/subject{subj}/mono.wav')  # currently 'trainset' is divided into train and test data
            fs, ambisonic = wavfile.read(f'{args.wav_base}/trainset/subject{subj}/ambisonic.wav')
            normalize = True
            reverb_pred = []
            # create audio files to be written as their errors are not currently calculated
            if key in args.test_points:
                for j in range(args.components):
                    reverb_pred.append(fftconvolve(mono, predicted_rir[j, :]))
                    if normalize and key in args.test_points:
                        with np.nditer(reverb_pred[j], op_flags=['readwrite']) as it:  # normalize based on dataset data
                            for x in it:
                                x[...] = x / max_val
                reverb_pred = np.array(reverb_pred).T

            # Filter and calculate error metrics for each frequency band
            for component in range(args.components):
                error_metrics[train_test][component]['spec_err'].append(np.abs(np.subtract(output[0], spec_data[0])).mean())
                # error_metrics[train_test]['mse_wav'].append(np.square(np.subtract(reverb_pred, ambisonic)).mean())  # todo check which is longer and slice

                rir_bands = np.tile(predicted_rir[component], (bands, 1)).T
                rir_bands_gt = np.tile(gt_rir[component], (bands, 1)).T
                filtered_pred = metrics.filter_rir(rir_bands, band_centerfreqs, fs)  # len, bands, pred and gt chan, len
                filtered_gt = metrics.filter_rir(rir_bands_gt, band_centerfreqs, fs)

                for band in range(bands):
                    error_metrics[train_test][component][band_centerfreqs[band]]['mse'].append(
                        np.square(np.subtract(filtered_pred[:, band], filtered_gt[:, band])).mean())
                    _, edc_db_pred = metrics.get_edc(filtered_pred[:, band])
                    rt60_pred = metrics.get_rt_from_edc(edc_db_pred, fs)
                    _, edc_db_gt = metrics.get_edc(filtered_gt[:, band])
                    rt60_gt = metrics.get_rt_from_edc(edc_db_gt, fs)
                    error_metrics[train_test][component][band_centerfreqs[band]]['rt60'].append(abs(rt60_gt - rt60_pred) / rt60_gt)  # todo: test with prm to 30 dB to match naf

                    delay = metrics.get_delay_samples(src_pos, rcv_pos)
                    drr_pred = 10 * np.log10(metrics.get_drr(filtered_pred[:, band], delay))
                    drr_gt = 10 * np.log10(metrics.get_drr(filtered_gt[:, band], delay))
                    error_metrics[train_test][component][band_centerfreqs[band]]['drr'].append(abs(drr_gt - drr_pred) / drr_gt)
                    c50_pred = 10 * np.log10(metrics.get_c50(filtered_pred[:, band], delay))
                    c50_gt = 10 * np.log10(metrics.get_c50(filtered_gt[:, band], delay))
                    error_metrics[train_test][component][band_centerfreqs[band]]['c50'].append(abs(c50_gt - c50_pred) / c50_gt)

                    # t = np.arange(len(edc_db_gt)) / fs
                    # plt.plot(t, edc_db_pred, label='Predicted EDC (dB)')
                    # plt.plot(t, edc_db_gt, label='Ground-truth EDC (dB)')
                    # plt.plot(t, np.ones(np.size(t)) * -60)
                    # plt.scatter(rt60_pred, -60, label='Predicted RT60')
                    # plt.scatter(rt60_gt, -60, label='GT RT60')
                    # plt.scatter(measure_rt60(predicted_rir[k], fs, 30), -60, label='Predicted RT60 PRA')
                    # plt.scatter(measure_rt60(gt_rir[k], fs, 30), -60, label='GT RT60 PRA')
                    # plt.title(f'Delay: {delay} samples ({src}-{rcv})')
                    # plt.legend()
                    # plt.show()

            # Plot some examples for checking the results
            if i < 1:
                plot_stft(output, spec_data, key)
                plot_wave(predicted_rir[0], gt_rir[0], key)
                # plot_wave(reverb_pred[:, 0], ambisonic[:, 0], key, 'audio waveform')
                # t = np.arange(len(edc_db_gt)) / fs
                # plt.plot(t, edc_db_pred, label='Predicted EDC (dB)')
                # plt.plot(t, edc_db_gt, label='Ground-truth EDC (dB)')
                # plt.plot(t, np.ones(np.size(t)) * -60)
                # plt.title(f'Delay: {delay} samples ({src}-{rcv})')
                # plt.legend()
                # plt.show()
            if key in args.test_points:
                plot_wave(predicted_rir[0], gt_rir[0], key)
                # plot_wave(wave_rir_out, ambisonic, key, 'audio waveform')
                pathlib.Path(args.wav_loc).mkdir(parents=True, exist_ok=True)
                if normalize:
                    wavfile.write(f'{args.wav_loc}/pred_{key}_s{subj}.wav', fs, reverb_pred.astype(np.float32))
                else:
                    wavfile.write(f'{args.wav_loc}/pred_{key}_s{subj}.wav', fs, reverb_pred.astype(np.int16))

    spec_obj.close()
    phase_obj.close()
    if write_errors:
        print_errors(error_metrics)
        pathlib.Path(args.metric_loc).mkdir(parents=True, exist_ok=True)
        with open(f'{args.metric_loc}/{options.error_file}.pkl', 'wb') as f:
            pickle.dump(error_metrics, f)


def to_wave(input_spec, hop_len, mean_val=None, std_val=None, gl=False, orig_phase=None):
    if mean_val is not None:
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
            out_wave = librosa.istft(f, hop_length=hop_len)
        else:
            out_wave = librosa.griffinlim(renorm_input, hop_length=hop_len, n_iter=40, momentum=0.5, random_state=64)
    else:
        f = renorm_input * (np.cos(orig_phase) + (1.j * np.sin(orig_phase)))
        out_wave = librosa.istft(f, win_length=400, hop_length=200)
    return out_wave


def to_wave_if(input_stft, input_if, hop_len):
    # 2 chanel input of shape [2,freq,time]
    # First input is logged mag
    # Second input is if divided by np.pi
    padded_input_stft = np.concatenate((input_stft, input_stft[:, -1:]), axis=1)
    padded_input_if = np.concatenate((input_if, input_if[:, -1:] * 0.0), axis=1)
    unwrapped = np.cumsum(padded_input_if, axis=-1) * np.pi
    phase_val = np.cos(unwrapped) + 1j * np.sin(unwrapped)
    restored = (np.exp(padded_input_stft) - 1e-3) * phase_val
    wave = librosa.istft(restored, hop_length=hop_len)
    return wave


if __name__ == '__main__':
    options = Options().parse()
    if options.test_points is not None and not options.recalculate_errors:
        print('Querying model at the wanted points for plotting and audio generation')
        test_model(options, options.test_points, False)
        if pathlib.Path(f'{options.metric_loc}/{options.error_file}.pkl').is_file():
            print(f'Loaded total errors from \'{options.error_file}\':')
            with open(f'{options.metric_loc}/{options.error_file}.pkl', 'rb') as f:
                print_errors(pickle.load(f))
    else:
        test_model(options)

    """
    # Where original RIR waveforms are stored and in which format
    originals = load_pkl('/worktmp/melandev/data/generated/rirs/order_0/room_10.0x6.0x2.5/grid_20x10/rt60_0.2/rirs.pickle')
    original = originals['0-199']
    
    """
