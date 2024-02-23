import librosa
import matplotlib.pyplot as plt
import numpy as np
import pickle


"""
    Various helper functions, mostly for plotting network query results (in test_query.py)
"""

METRICS_BAND = ['mse', 'rt60', 'drr', 'c50', 'errors']  # add spectral error for bands?
METRICS_CHANNEL = ['spec_err_', 'mse_', 'rt60_', 'drr_', 'c50_', 'mse_wav']
METRICS_DIRECTIONAL = ['amb_e', 'amb_edc', 'dir_rir', 'ild', 'icc']


def load_pkl(path):
    with open(path, 'rb') as loaded_pkl_obj:
        loaded_pkl = pickle.load(loaded_pkl_obj)
    return loaded_pkl


def plot_errors(full_data, error):
    centerfreqs = [125, 250, 500, 1000, 2000, 4000]
    labels = ['125 Hz', '250 Hz', '500 Hz', '1000 Hz', '2000 Hz', '4000 Hz', 'broadband']
    plt.xticks(np.arange(7), labels)
    for method in ['off-grid mono', 'off-grid omni channel', 'on-grid mono', 'on-grid omni channel']:
        train_test = ['train', 'test'] if 'off-grid' not in method else ['train']
        for types in train_test:
            errors = []
            for i, freq in enumerate(centerfreqs):
                errors.append(np.mean(full_data[method][types][0][freq][error]))
            errors.append(np.mean(full_data[method][types][0][f'{error}_']))  # broadband
            label = f'{method} {types}' if 'test' in train_test else f'{method}'
            plt.plot(labels, errors, 'o-', label=label)
    plt.ylabel(f'{error.upper()}')
    plt.legend()
    plt.show()


def plot_stft_log(pred, gt, points, n_fft=128, fs=16000):  # tmp
    fig, ax = plt.subplots()
    img = librosa.display.specshow(pred[0, 0], y_axis='log', x_axis='time', ax=ax, sr=fs, hop_length=n_fft // 2, shading='nearest')
    ax.set_title('Power spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")


def plot_stft(pred, gt, points, n_fft=128, fs=16000):
    t = np.arange(pred.shape[-1]) / fs * n_fft / 2  # 50% overlap, parametrize?
    f = np.arange(pred.shape[-2]) / (n_fft / fs)
    fig, axarr = plt.subplots(1, 2)
    # fig.suptitle(f'Predicted log impulse response {points}', fontsize=16)  # comment out for final plots
    plot = axarr[0].pcolormesh(t, f, pred[0, 0], vmin=np.min(gt) * 1.1, vmax=np.max(gt) * 0.9)
    axarr[0].set_title(f'Prediction ({points.replace("_", "─")})')
    plot2 = axarr[1].pcolormesh(t, f, gt[0, 0], vmin=np.min(gt) * 1.1, vmax=np.max(gt) * 0.9)
    axarr[1].set_title(f'Ground-truth ({points.replace("_", "─")})')
    # plot3 = axarr[2].pcolormesh(t, f, gt[0, 0] - pred[0, 0], vmin=np.min(gt) * 1.1, vmax=np.max(gt) * 0.9)
    # axarr[2].set_title('Error')
    plt.setp(axarr, xlabel='Time (s)'), plt.setp(axarr, ylabel='Frequency (Hz)')
    fig.colorbar(plot, format='%+2.f dB'), fig.colorbar(plot2, format='%+2.f dB')  # , fig.colorbar(plot3, format='%+2.f dB')
    plt.show()


def plot_stft_ambi(pred, gt, points, n_fft=128, fs=16000):
    t = np.arange(pred.shape[-1]) / fs * n_fft / 2  # 50% overlap, parametrize?
    f = np.arange(pred.shape[-2]) / (n_fft / fs)
    channels = ['W', 'Y', 'Z', 'X']

    plt.rcParams.update({'font.size': 16})
    fig = plt.figure()
    subfigs = fig.subfigures(2, 2)
    for channel, subfig in enumerate(subfigs.flat):
        subfig.suptitle(f'{channels[channel]} channel')
        axs = subfig.subplots(1, 2)
        plot1 = axs[0].pcolormesh(t, f, pred[0, channel], vmin=np.min(gt) * 1.1, vmax=np.max(gt) * 0.9)
        axs[0].set_title(f'Prediction ({points.replace("_", "─")})')
        plot2 = axs[1].pcolormesh(t, f, gt[0, channel], vmin=np.min(gt) * 1.1, vmax=np.max(gt) * 0.9)
        axs[1].set_title(f'Ground-truth ({points.replace("_", "─")})')
        plt.setp(axs[0], xlabel='Time (s)'), plt.setp(axs[0], ylabel='Frequency (Hz)')
        plt.setp(axs[1], xlabel='Time (s)'), plt.setp(axs[1], ylabel='Frequency (Hz)')
        fig.colorbar(plot1, format='%+2.f dB'), fig.colorbar(plot2, format='%+2.f dB')
    plt.show()
    plt.rcParams.update({'font.size': 22})


def plot_stft_ph(pred, gt, points, n_fft=128, fs=16000):
    t = np.arange(pred.shape[-1]) / fs * n_fft / 2  # 50% overlap, parametrize?
    f = np.arange(pred.shape[-2]) / (n_fft / fs)
    fig, axarr = plt.subplots(1, 2)
    # fig.suptitle(f'Predicted log impulse response {points}', fontsize=16)  # comment out for final plots
    plot = axarr[0].pcolormesh(t, f, pred[0, 0], vmin=np.min(gt) * 1.1, vmax=np.max(gt) * 0.9)
    axarr[0].set_title(f'Prediction ({points.replace("_", "─")})')
    plot2 = axarr[1].pcolormesh(t, f, gt[0, 0], vmin=np.min(gt) * 1.1, vmax=np.max(gt) * 0.9)
    axarr[1].set_title(f'Ground-truth ({points.replace("_", "─")})')
    # plot3 = axarr[2].pcolormesh(t, f, gt[0, 0] - pred[0, 0], vmin=np.min(gt) * 1.1, vmax=np.max(gt) * 0.9)
    # axarr[2].set_title('Error')
    plt.setp(axarr, xlabel='Time (s)'), plt.setp(axarr, ylabel='Frequency (Hz)')
    fig.colorbar(plot, format='%+2.2f rad'), fig.colorbar(plot2, format='%+2.2f rad')  # , fig.colorbar(plot3, format='%+2.f dB')
    plt.show()


def plot_stft_imshow(pred, gt, points):
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


def plot_wave(pred, gt, points, sr=16000):
    t = np.arange(len(gt)) / sr
    axes = plt.axes()
    axes.set_xlim([0, 0.08])
    axes.set_ylim([min(gt) * 1.1, max(gt) * 1.1])
    plt.plot(t, gt, label=f'Ground-truth ({points.replace("_", "─")})', color='k', alpha=1.0)
    plt.plot(t, pred, label=f'Prediction ({points.replace("_", "─")})', color='mediumseagreen', alpha=0.8)
    plt.ylabel('Amplitude')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.show()


def plot_wave_ambi(pred, gt, points, sr=16000):
    t = np.arange(len(gt[0])) / sr
    channels = ['W', 'Y', 'Z', 'X']

    # plt.rcParams.update({'font.size': 16})
    fig, axarr = plt.subplots(2, 2)
    for channel, subfig in enumerate(axarr.flat):
        subfig.set_title(f'{channels[channel]} channel')
        subfig.set_xlim([0, 0.08])
        subfig.set_ylim([np.min(gt) * 1.1, np.max(gt) * 1.1])
        subfig.plot(t, gt[channel], label=f'Ground-truth ({points.replace("_", "─")})', color='k', alpha=1.0)
        subfig.plot(t, pred[channel], label=f'Prediction ({points.replace("_", "─")})', color='mediumseagreen', alpha=0.8)
        subfig.set_ylabel('Amplitude')
        subfig.set_xlabel('Time (s)')
        subfig.legend()
    plt.show()
    plt.rcParams.update({'font.size': 22})


def plot_wave_err(pred, gt, points, name='impulse response', sr=16000):
    fig, axarr = plt.subplots(2, 1)
    # fig.suptitle(f'Predicted vs. GT {name} between {points}', fontsize=16)  # comment out for final plots
    max_len = max(np.arange(len(gt)) / sr)
    axarr[0].plot(np.arange(len(pred)) / sr, pred)
    axarr[0].set_xlim([0, max_len])
    axarr[0].set_ylim([None, max(gt) * 1.1])
    axarr[0].set_title(f'Prediction ({points.replace("_", "─")})')
    axarr[1].plot(np.arange(len(gt)) / sr, gt)
    axarr[1].set_xlim([0, max_len])
    axarr[1].set_ylim([None, max(gt) * 1.1])
    axarr[1].set_title(f'Ground-truth ({points.replace("_", "─")})')
    # axarr[2].plot(np.arange(len(np.subtract(pred[:len(gt)], gt))) / sr, np.subtract(pred[:len(gt)], gt))
    # axarr[2].set_ylim([None, max(gt) * 1.1])
    # axarr[2].set_xlim([0, max_len])
    # axarr[2].set_title('Error')
    plt.ylabel('Amplitude')
    plt.xlabel('Time (s)')
    plt.show()


def print_errors(error_metrics):  # train, channel, band, metric
    global METRICS_CHANNEL
    if 'directional' in error_metrics:  # directional, train, metric
        print('Directional error metrics')
        for train_test in ['train', 'test']:
            print(f'  Directional errors for {train_test} points')
            for metric, data in error_metrics['directional'][train_test].items():
                if metric == 'dir_rir':
                    print('    Directed RIR errors')
                    for submetric, value in data.items():
                        print(f'      avg. {submetric}: {np.average(value):.9f}')
                else:
                    print(f'    avg. {metric}: {np.average(data):.9f}')
    for train_test in ['train', 'test']:
        print(f'Errors for {train_test} points')
        for channel in error_metrics[train_test]:
            print(f'  channel {channel}')
            # if len(error_metrics[train_test][channel]["mse_wav"]) > 0:  # add if necessary
            for data in error_metrics[train_test][channel]:
                if data in METRICS_CHANNEL and data != 'mse_wav':
                    print(f'    avg. channel {data.upper()[:-1]}: {np.average(error_metrics[train_test][channel][data]):.9f}')
                else:
                    print(f'    band: {data}')
                    for metric in error_metrics[train_test][channel][data]:
                        if len(error_metrics[train_test][channel][data][metric]) > 0:
                            print(f'      avg. {metric.upper()}: {np.average(error_metrics[train_test][channel][data][metric]):.9f}')


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
    off_grid = load_pkl(f'./out/ambisonics_0_4x2/metrics/errors2.pkl')
    off_grid_sh = load_pkl(f'./out/ambisonics_1_4x2/metrics/errors2.pkl')
    mono = load_pkl(f'./out/ambisonics_0_20x10/metrics/errors2.pkl')
    sh = load_pkl(f'./out/ambisonics_1_20x10/metrics/errors2.pkl')
    full = {'off-grid mono': off_grid, 'off-grid omni channel': off_grid_sh, 'on-grid mono': mono, 'on-grid omni channel': sh}

    plot_errors(full, 'mse')
    plot_errors(full, 'c50')
    plot_errors(full, 'drr')
    plot_errors(full, 'rt60')
