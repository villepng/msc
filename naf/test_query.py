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
import naf.utils as utl

from pyroomacoustics.experimental.rt60 import measure_rt60
from scipy.io import wavfile
from scipy.signal import fftconvolve
from torchview import draw_graph

from naf.model.modules import EmbeddingModuleLog
from naf.model.networks import KernelResidualFCEmbeds
from naf.options import Options


METRICS_BAND = ['mse', 'rt60', 'drr', 'c50', 'edc', 'errors']
METRICS_BINAURAL = ['ild', 'icc', 'ild_pred', 'ild_gt', 'icc_pred', 'icc_gt']
METRICS_CHANNEL = ['spec_err_', 'mse_', 'rt60_', 'drr_', 'c50_', 'edc_', 'mse_wav']
METRICS_DIRECTIONAL = ['amb_e', 'amb_edc', 'dir_rir', 'binaural']

RNG = np.random.default_rng(0)


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


def load_gt_data(args):
    spec_obj = h5py.File(f'{args.spec_base}/{args.apt}.h5', 'r')
    phase_obj = h5py.File(f'{args.phase_base}/{args.apt}.h5', 'r')
    with open(f'metadata/ambisonics_{args.order}_{args.grid}/normalization/{args.apt}_max_val.txt', 'r') as f:
        max_val = f.readlines()

    with open(f'{args.coor_base}/{args.apt}/points.txt', 'r') as f:
        lines = f.readlines()
    points = [x.replace('\n', '').split(' ') for x in lines]
    positions = dict()
    for row in points:
        readout = [float(xyz) for xyz in row[1:]]
        positions[row[0]] = [readout[0], readout[1]]

    train_test_split = utl.load_pkl(f'{args.split_loc}/{args.apt}_complete.pkl')
    train_keys = train_test_split[0]
    test_keys = train_test_split[1]

    return spec_obj, phase_obj, positions, train_keys, test_keys, float(max_val[0])


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


def test_model(args, test_points=None, write_errors=True):
    global RNG
    apt = args.apt
    max_len = args.max_len[apt]
    weight_path = f'{args.model_save_loc}/{apt}/0200.chkpt'
    min_max = utl.load_pkl(f'{args.minmax_base}/{args.apt}_minmax.pkl')
    min_pos, max_pos = np.array(min_max[0][0:2]), np.array(min_max[1][0:2])
    output_device = 0
    orientation = 0

    # Load mean and std data & gt data and prepare the network
    mean_std = utl.load_pkl(f'{args.mean_std_base}/{apt}.pkl')
    mean = torch.from_numpy(mean_std[0]).float()
    std = 3.0 * torch.from_numpy(mean_std[1]).float()
    # std_phase = 3.0 * torch.from_numpy(mean_std[2]).float()
    spec_obj, phase_obj, points, train_keys, test_keys, max_val = load_gt_data(args)
    if test_points is not None:
        train_keys[orientation] = []
        test_keys[orientation] = test_points
    if args.grid != args.model_save_loc[-len(args.grid):]:  # only test data for off-grid points
        test_keys[orientation].extend(train_keys[orientation])
        train_keys[orientation] = []
    network = prepare_network(weight_path, args, output_device, min_pos, max_pos)

    # Polling the network to calculate error metrics
    band_centerfreqs = np.array([125, 250, 500, 1000, 2000, 4000])
    bands = len(band_centerfreqs)
    wave_mse_window = 50 * args.sr // 1000  # Calculate waveform MSE for 50ms after direct sound
    cutoff = int(0.29 * args.sr)  # cutoff for edc error
    error_metrics = utl.get_error_metric_dict(args.components, band_centerfreqs)  # train, channel, band, metrics
    fs = args.sr

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
                # model_graph = draw_graph(network, input_size=(1, 5376, 126), graph_dir='TB', depth=6, device='cuda:0')  # input_data=(net_input, degree, non_norm_position.squeeze(1))
                # model_graph.visual_graph.render(format='pdf')
                output = network(net_input, non_norm_position.squeeze(1)).squeeze(3).transpose(1, 2)
            # phase = output[:, :, :, 1]
            # output = output[:, :, :, 0]
            # phase = (output.reshape(1, args.components, args.freq_bins, max_len).cpu() * std_phase).numpy()
            output = (output.reshape(1, args.components, args.freq_bins, max_len).cpu() * std + mean).numpy()

            ''''# Random phase reconstruction per image2reverb for later parts of the RIR
            full_phase = np.zeros([1, args.components, args.freq_bins, max_len])
            full_phase[:, :, :, :13] = phase[:, :, :, :13]
            np.random.seed(1234)
            rp = np.random.uniform(-np.pi, np.pi, [1, args.components, args.freq_bins, max_len - 13])
            full_phase[:, :, :, 13:] = rp'''

            # Convert into time domain to calculate most metrics
            # predicted_rir = utl.to_wave_if(output[0], phase_data[0], args.hop_len)  # using original phases
            predicted_rir = utl.to_wave(output[0], args.hop_len)  # [channels, length], random phase
            # predicted_rir = utl.to_wave_if(output[0], phase[0], args.hop_len)  # predicted phase
            # predicted_rir = utl.to_wave_if(spec_data[0], phase[0], args.hop_len)  # predicted and random phase
            gt_rir = utl.to_wave_if(spec_data[0], phase_data[0], args.hop_len)  # could also load original RIR, but shouldn't matter
            # utl.plot_stft(phase, phase_data, key)
            # gt_rir = utl.to_wave(spec_data[0], args.hop_len)  # test reconstructing GT with random phase
            # t = np.arange(len(gt_rir)) / 16000
            # plt.plot(t, 20*np.log10(abs(gt_rir)))
            # plt.show()

            # Convert from src and rcv points into 'subjects' in the original dataset format
            src, rcv = int(src), int(rcv)
            if rcv < src:
                subj = src * args.subj_offset + rcv + 1
            else:
                subj = src * args.subj_offset + rcv
            normalize = True
            reverb_pred = []
            # create audio files to be written as their errors are not currently calculated
            if key in args.test_points:
                fs, mono = wavfile.read(f'{args.wav_base}/trainset/subject{subj}/mono.wav')  # currently 'trainset' is divided into train and test data
                fs, ambisonic = wavfile.read(f'{args.wav_base}/trainset/subject{subj}/ambisonic.wav')
                for j in range(args.components):
                    reverb_pred.append(fftconvolve(mono, predicted_rir[j, :]))
                    if normalize and key in args.test_points:
                        with np.nditer(reverb_pred[j], op_flags=['readwrite']) as it:  # normalize based on dataset data
                            for x in it:
                                x[...] = x / max_val
                reverb_pred = np.array(reverb_pred).T

            # Calculate ambisonic error metrics
            delay = metrics.get_delay_samples(src_pos, rcv_pos)
            win_end = delay + wave_mse_window
            if args.components > 1:
                error_metrics['directional'][train_test]['amb_e'].append(metrics.get_ambisonic_energy_err(predicted_rir, gt_rir))
                error_metrics['directional'][train_test]['amb_edc'].append(metrics.get_ambisonic_edc_err(predicted_rir, gt_rir))
                metrics.calculate_directed_rir_errors(predicted_rir, gt_rir, RNG, delay, error_metrics, train_test, src_pos, rcv_pos)
                metrics.calculate_binaural_error_metrics(predicted_rir, gt_rir, RNG, error_metrics, train_test, src, rcv, order=int(args.order))

            if True:  # and key in args.test_points
                # Filter and calculate error metrics
                for component in range(args.components):  # 'spec_err_', 'mse_', 'rt60_', 'drr_', 'c50_'
                    # Overall error metrics for each component
                    error_metrics[train_test][component]['spec_err_'].append(np.abs(np.subtract(output[:, component], spec_data[:, component])).mean())
                    error_metrics[train_test][component]['mse_'].append(np.square(np.subtract(predicted_rir[component, delay:win_end], gt_rir[component, delay:win_end])).mean())
                    # error_metrics[train_test]['mse_wav'].append(np.square(np.subtract(reverb_pred, ambisonic)).mean())  # todo check which is longer and slice
                    _, edc_db_pred = metrics.get_edc(predicted_rir[component])
                    rt60_pred, _ = metrics.get_rt_from_edc(edc_db_pred, fs)
                    _, edc_db_gt = metrics.get_edc(gt_rir[component])
                    rt60_gt, _ = metrics.get_rt_from_edc(edc_db_gt, fs)
                    error_metrics[train_test][component]['rt60_'].append(np.abs(rt60_gt - rt60_pred) / rt60_gt)
                    error_metrics[train_test][component]['edc_'].append(np.abs(edc_db_pred[:cutoff] - edc_db_gt[:cutoff]).mean())

                    drr_pred = 10 * np.log10(metrics.get_drr(predicted_rir[component], delay))
                    drr_gt = 10 * np.log10(metrics.get_drr(gt_rir[component], delay))
                    error_metrics[train_test][component]['drr_'].append(np.abs(drr_gt - drr_pred))
                    c50_pred = 10 * np.log10(metrics.get_c50(predicted_rir[component], delay))
                    c50_gt = 10 * np.log10(metrics.get_c50(gt_rir[component], delay))
                    error_metrics[train_test][component]['c50_'].append(np.abs(c50_gt - c50_pred))

                    '''t = np.arange(len(edc_db_gt)) / fs
                    plt.plot(t, edc_db_pred, label='Predicted EDC (dB)')
                    plt.plot(t, edc_db_gt, label='Ground-truth EDC (dB)')
                    plt.plot(t, np.ones(np.size(t)) * -60)
                    plt.vlines(0.29, 0, -120)
                    plt.scatter(rt60_pred, -60, label='Predicted RT60')
                    plt.scatter(rt60_gt, -60, label='GT RT60')
                    # measure_rt60(predicted_rir[component], fs, 60, True)
                    # measure_rt60(predicted_rir[component], fs, 30, True)
                    # plt.scatter(measure_rt60(predicted_rir[component], fs, 60), -60, label='Predicted RT60 PRA')
                    # plt.scatter(measure_rt60(gt_rir[component], fs, 60), -60, label='GT RT60 PRA')
                    plt.title(f'Delay: {delay / fs:.3f}s ({src}-{rcv}), ({component})')
                    plt.legend()
                    plt.show()'''

                    # Filtering
                    rir_bands = np.tile(predicted_rir[component], (bands, 1)).T
                    rir_bands_gt = np.tile(gt_rir[component], (bands, 1)).T
                    filtered_pred = metrics.filter_rir(rir_bands, band_centerfreqs, fs)  # len, bands, pred and gt chan, len
                    filtered_gt = metrics.filter_rir(rir_bands_gt, band_centerfreqs, fs)

                    # plot_wave(predicted_rir[component], gt_rir[component], f'{src}-{rcv}, ch{component}')
                    # Error metrics for each frequency band
                    for band in range(bands):
                        # utl.plot_wave(filtered_pred[:, band], filtered_gt[:, band], f'{src}-{rcv}, {component}-{band}')
                        error_metrics[train_test][component][band_centerfreqs[band]]['mse'].append(np.square(np.subtract(filtered_pred[delay:win_end, band], filtered_gt[delay:win_end, band])).mean())
                        _, edc_db_pred = metrics.get_edc(filtered_pred[:, band])
                        if band == 0:
                            offset, interval = (1, 10)
                        elif band == 1:
                            offset, interval = (5, 15)
                        elif band == 2:
                            offset, interval = (5, 20)
                        else:
                            offset, interval = (5, 30)
                        rt60_pred, a1 = metrics.get_rt_from_edc(edc_db_pred, fs, offset, interval)
                        _, edc_db_gt = metrics.get_edc(filtered_gt[:, band])
                        rt60_gt, a2 = metrics.get_rt_from_edc(edc_db_gt, fs, offset, interval)
                        error_metrics[train_test][component][band_centerfreqs[band]]['rt60'].append(np.abs(rt60_gt - rt60_pred) / rt60_gt)
                        error_metrics[train_test][component][band_centerfreqs[band]]['edc'].append(np.abs(edc_db_pred[:cutoff] - edc_db_gt[:cutoff]).mean())
                        '''t = np.arange(len(edc_db_gt)) / fs
                        t2 = np.arange(rt60_pred * fs) / fs
                        plt.plot(t2, a1 * t2, label='pred')
                        t3 = np.arange(rt60_gt * fs) / fs
                        plt.plot(t3, a2 * t3, label='gt')

                        plt.plot(t, edc_db_pred, label='Predicted EDC (dB)')
                        plt.plot(t, edc_db_gt, label='Ground-truth EDC (dB)')
                        plt.scatter(rt60_pred, -60, label='Predicted RT60')
                        plt.scatter(rt60_gt, -60, label='GT RT60')
                        # measure_rt60(filtered_pred[:, band], fs, 30, True)
                        # measure_rt60(filtered_gt[:, band], fs, 30, True)
                        plt.title(f'Delay: {delay / fs:.3f}s ({src}-{rcv}, {band_centerfreqs[band]} Hz:{component}-{band})')
                        plt.legend()
                        plt.show()'''

                        drr_pred = 10 * np.log10(metrics.get_drr(filtered_pred[:, band], delay))
                        drr_gt = 10 * np.log10(metrics.get_drr(filtered_gt[:, band], delay))
                        error_metrics[train_test][component][band_centerfreqs[band]]['drr'].append(np.abs(drr_gt - drr_pred))
                        c50_pred = 10 * np.log10(metrics.get_c50(filtered_pred[:, band], delay))
                        c50_gt = 10 * np.log10(metrics.get_c50(filtered_gt[:, band], delay))
                        error_metrics[train_test][component][band_centerfreqs[band]]['c50'].append(np.abs(c50_gt - c50_pred))

            # Plot some examples for checking the results
            if i < 1 and False:
                utl.plot_stft(output, spec_data, key)
                utl.plot_wave(predicted_rir[0], gt_rir[0], key)
                # utl.plot_wave(reverb_pred[:, 0], ambisonic[:, 0], key, 'audio waveform')
                # t = np.arange(len(edc_db_gt)) / fs
                # plt.plot(t, edc_db_pred, label='Predicted EDC (dB)')
                # plt.plot(t, edc_db_gt, label='Ground-truth EDC (dB)')
                # plt.plot(t, np.ones(np.size(t)) * -60)
                # plt.title(f'Delay: {delay} samples ({src}-{rcv})')
                # plt.legend()
                # plt.show()
            if key in args.test_points and False:
                '''with open(f'./out/tmp/{key}.pkl', 'wb') as f:
                    pickle.dump(predicted_rir, f)
                with open(f'./out/tmp/{key}_gt.pkl', 'wb') as f:
                    pickle.dump(gt_rir, f)
                with open(f'./out/tmp/{key}_stft.pkl', 'wb') as f:
                    pickle.dump(output, f)'''
                utl.plot_stft_ambi(output, spec_data, key)
                '''from naf.data_loading.data_maker import if_compute
                np.random.seed(1234)
                rp = np.random.uniform(-np.pi, np.pi, output[0].shape)
                rnd_ph = output * (np.cos(rp) + (1.j * np.sin(rp)))
                gen_if = if_compute(np.angle(rnd_ph)) / np.pi
                plot_stft_ph(gen_if, phase_data, key)'''
                utl.plot_wave_ambi(predicted_rir, gt_rir, key)
                # plot_wave(wave_rir_out, ambisonic, key, 'audio waveform')
                pathlib.Path(args.wav_loc).mkdir(parents=True, exist_ok=True)
                if normalize:
                    wavfile.write(f'{args.wav_loc}/pred_{key}_s{subj}.wav', fs, reverb_pred.astype(np.float32))
                else:
                    wavfile.write(f'{args.wav_loc}/pred_{key}_s{subj}.wav', fs, reverb_pred.astype(np.int16))

    spec_obj.close()
    phase_obj.close()
    if write_errors:
        pathlib.Path(args.metric_loc).mkdir(parents=True, exist_ok=True)
        with open(f'{args.metric_loc}/{options.error_file}.pkl', 'wb') as f:
            pickle.dump(error_metrics, f)
        utl.print_errors(error_metrics)


if __name__ == '__main__':
    plt.rcParams.update({'font.size': 22})
    # plt.rc('axes', titlesize=16)
    # plt.rc('legend', fontsize=16)
    options = Options().parse()
    if options.test_points is not None and not options.recalculate_errors:
        print('Querying model at the wanted points for plotting and audio generation')
        test_model(options, options.test_points, False)
        if pathlib.Path(f'{options.metric_loc}/{options.error_file}.pkl').is_file():
            print(f'Loaded total errors from \'{options.error_file}\':')
            with open(f'{options.metric_loc}/{options.error_file}.pkl', 'rb') as f:
                utl.print_errors(pickle.load(f))
    else:
        test_model(options)

    """
    # Where original RIR waveforms are stored and in which format
    originals = load_pkl('/worktmp/melandev/data/generated/rirs/order_0/room_10.0x6.0x2.5/grid_20x10/rt60_0.2/rirs.pickle')
    original = originals['0-199']
    
    """
