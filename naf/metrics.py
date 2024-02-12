import matplotlib.pyplot as plt
import numpy as np
import scipy
import spaudiopy as spa

# from naf.data_loading.data_maker import GetSpec


def calculate_directed_rir_errors(pred_rir, gt_rir, delay, error_metrics, train_test, fs=16000):
    """ Currently only works with 1st order ambisonics
    :param pred_rir: [chn, len]
    :param gt_rir: [chn, len]
    :param delay: sound travel time in samples
    :param error_metrics: where to save errors
    :param train_test: save to train or test part
    :param fs: sample rate
    :return: None
    """
    if pred_rir.shape[0] != 4 or gt_rir.shape[0] != 4:
        raise NotImplementedError('RIRs must be 1st order ambisonics')
    # elevation = 0  # currently not used
    azimuth = np.pi / 4  # todo: random with set seed, certain angles?
    beamer = np.array([1, np.sin(azimuth) * 1, 0, np.cos(azimuth) * 1])
    dir_rir_pred = np.zeros([pred_rir.shape[-1]])
    dir_rir_gt = np.zeros([pred_rir.shape[-1]])
    for channel in range(pred_rir.shape[0]):
        dir_rir_pred += pred_rir[channel] * beamer[channel]
        dir_rir_gt += gt_rir[channel] * beamer[channel]

    # Calculate "normal" metrics for directer RIRs
    error_metrics['directional'][train_test]['dir_rir']['mse'].append(np.square(dir_rir_pred - dir_rir_gt).mean())
    _, edc_db_pred = get_edc(dir_rir_pred)
    rt60_pred = get_rt_from_edc(edc_db_pred, fs)
    _, edc_db_gt = get_edc(dir_rir_gt)
    rt60_gt = get_rt_from_edc(edc_db_gt, fs)
    error_metrics['directional'][train_test]['dir_rir']['rt60'].append(np.abs(rt60_gt - rt60_pred) / rt60_gt)

    drr_pred = 10 * np.log10(get_drr(dir_rir_pred, delay))
    drr_gt = 10 * np.log10(get_drr(dir_rir_gt, delay))
    error_metrics['directional'][train_test]['dir_rir']['drr'].append(np.abs(drr_gt - drr_pred) / drr_gt)
    c50_pred = 10 * np.log10(get_c50(dir_rir_pred, delay))
    c50_gt = 10 * np.log10(get_c50(dir_rir_gt, delay))
    error_metrics['directional'][train_test]['dir_rir']['c50'].append(np.abs(c50_gt - c50_pred) / c50_gt)


def filter_rir(rir, f_center, fs):
    bands = len(f_center)
    order = 1000
    filters = np.zeros((order + 1, bands))
    for i in range(bands):
        if i == 0:
            fl = 30.
            fh = np.sqrt(f_center[i] * f_center[i + 1])
            wl = fl / (fs / 2.)
            wh = fh / (fs / 2.)
            filters[:, i] = scipy.signal.firwin(order + 1, [wl, wh], pass_zero='bandpass')
        elif i == bands - 1:
            fl = np.sqrt(f_center[i] * f_center[i - 1])
            w = fl / (fs / 2.)
            filters[:, i] = scipy.signal.firwin(order + 1, w, pass_zero='highpass')
        else:
            fl = np.sqrt(f_center[i] * f_center[i - 1])
            fh = np.sqrt(f_center[i] * f_center[i + 1])
            wl = fl / (fs / 2.)
            wh = fh / (fs / 2.)
            filters[:, i] = scipy.signal.firwin(order + 1, [wl, wh], pass_zero='bandpass')

    temp_rir = np.append(rir, np.zeros((order, bands)), axis=0)
    rir_filt = scipy.signal.fftconvolve(filters, temp_rir, axes=0)[:temp_rir.shape[0], :]

    # import matplotlib.pyplot as plt
    # # plt.plot(filters)
    # # plt.show()
    # spec_getter = GetSpec(components=6)
    # real_spec, img_spec, raw_phase = spec_getter.transform(rir_filt.T)
    # f, axarr = plt.subplots(3, 2)
    # axarr[0, 0].imshow(real_spec[0])
    # axarr[0, 1].imshow(real_spec[1])
    # axarr[1, 0].imshow(real_spec[2])
    # axarr[1, 1].imshow(real_spec[3])
    # axarr[2, 0].imshow(real_spec[4])
    # axarr[2, 1].imshow(real_spec[5])
    # f.show()

    return rir_filt[500:-1, :]  # remove filtering delay


def get_ambisonic_edc_err(pred_rir, gt_rir, cutoff=0.29, fs=16000):
    """
    :param pred_rir: ambisonic rir as [chn, len]
    :param gt_rir: ambisonic rir as [chn, len]
    :param cutoff: only use edc until this point (seconds), as it decays too fast afterwards
    :param fs: sample rate of the rirs
    :return:
    """
    cutoff = int(cutoff * fs)  # 0.29s seems fine for the most part
    edc_err = 0
    for channel in range(pred_rir.shape[0]):
        _, edc_db_pred = get_edc(pred_rir[channel])
        _, edc_db_gt = get_edc(gt_rir[channel])
        edc_err += np.abs(edc_db_pred[:cutoff] - edc_db_gt[:cutoff])
    return np.mean(edc_err)


def get_ambisonic_energy_err(pred_rir, gt_rir):
    """
    :param pred_rir: ambisonic rir as [chn, len]
    :param gt_rir: ambisonic rir as [chn, len]
    :return:
    """
    e_pred = 0
    e_gt = 0
    for channel in range(pred_rir.shape[0]):
        e_pred += np.square(pred_rir[channel])
        e_gt += np.square(gt_rir[channel])
    return np.mean(np.abs(e_pred - e_gt))


def get_binaural_error_metrics(pred_rir, gt_rir, rng, fs=16000, order=1):
    """
    :param pred_rir: ambisonic rir as [chn, len]
    :param gt_rir: ambisonic rir as [chn, len]
    :param rng: random number generator with 0 seed
    :param fs: sample rate of the rirs
    :param order: ambisonics order used
    :return: inter-channel level difference error and interchannel coherence error
    """
    hrir = spa.io.load_sofa_hrirs(f'../../data/hrir/mit_kemar_normal_pinna.sofa')  # parameter todo
    hrir = spa.decoder.magls_bin(hrir, order)
    brir_pred = spa.decoder.sh2bin(pred_rir, hrir)
    brir_gt = spa.decoder.sh2bin(gt_rir, hrir)
    white_noise = rng.standard_normal(1 * fs)
    bin_pred, bin_gt = [], []
    for i in range(2):
        bin_pred.append(scipy.signal.fftconvolve(white_noise, brir_pred[i]))
        bin_gt.append(scipy.signal.fftconvolve(white_noise, brir_gt[i]))

    '''from scipy.io import wavfile
    white_noise = white_noise / max(white_noise)
    bin_pred, bin_gt = np.array(bin_pred), np.array(bin_gt)
    bin_pred, bin_gt = bin_pred / np.max(bin_pred), bin_gt / np.max(bin_gt)
    wavfile.write(f'../../data/tmp/mono.wav', fs, white_noise.astype(np.float32))
    wavfile.write(f'../../data/tmp/bin_pred.wav', fs, np.array(bin_pred).astype(np.float32).T)
    wavfile.write(f'../../data/tmp/bin_gt.wav', fs, np.array(bin_gt).astype(np.float32).T)'''

    e_pred_l, e_pred_r, e_gt_l, e_gt_r = np.sum(np.square(bin_pred[1])), np.sum(np.square(bin_pred[0])), np.sum(np.square(bin_gt[1])), np.sum(np.square(bin_gt[0]))

    ild_err = 10 * np.log10(e_pred_l / e_pred_r) - 10 * np.log10(e_gt_l / e_gt_r)
    icc_err = np.sum(bin_pred[1] * bin_pred[0]) / np.sqrt(e_pred_l * e_pred_r) - np.sum(bin_gt[1] * bin_gt[0]) / np.sqrt(e_gt_l * e_gt_r)

    return np.abs(ild_err), np.abs(icc_err)


def get_c50(rir, delay, fs=16000):
    l_5ms = int(0.005 * fs)
    start = max(delay - l_5ms, 0)
    early = 0
    late = 0
    for i in range(start, delay + l_5ms * 10):
        early += rir[i] ** 2
    for i in range(delay + l_5ms * 10, len(rir)):
        late += rir[i] ** 2

    # t = np.arange(len(rir)) / fs
    # plt.plot(t, rir, alpha=0.5)
    # plt.scatter(t[start], rir[start], s=100, c='green', marker='x')
    # plt.scatter(t[delay + l_5ms * 10], rir[delay + l_5ms * 10], s=100, c='red', marker='x')
    # plt.show()

    return early / late


def get_delay_samples(src, rcv, fs=16000, v=343):
    distances = ((np.array(src) - np.array(rcv)) ** 2) ** (1 / 2)

    return int(np.sum(distances) / v * fs)


def get_drr(rir, delay, fs=16000):
    l_5ms = int(0.005 * fs)
    start = max(delay - l_5ms, 0)

    early = 0
    late = 0
    for i in range(start, delay + l_5ms):
        early += rir[i] ** 2
    for i in range(delay + l_5ms, len(rir)):
        late += rir[i] ** 2

    # t = np.arange(len(rir)) / fs
    # plt.plot(t, rir, alpha=0.5)
    # plt.scatter(t[start], rir[start], s=100, c='green', marker='x')
    # plt.scatter(t[delay + l_5ms], rir[delay + l_5ms], s=100, c='red', marker='x')
    # plt.title(delay)
    # plt.show()

    return early / late


def get_edc(rir, normalize=True):
    """ Compute the inverse cumulative sum of the squared RIR (that's the EDC)
    :param rir: single-channel rir
    :param normalize:
    :return:
    """
    rir2_flipped = np.flip(rir ** 2)
    rir2_flipped_csum = np.zeros(np.size(rir))
    rir2_flipped_csum[0] = rir2_flipped[0]
    for n in range(1, len(rir)):
        rir2_flipped_csum[n] = rir2_flipped_csum[n - 1] + rir2_flipped[n]
    edc = np.flip(rir2_flipped_csum)
    if normalize:
        edc = edc / edc[0]
    with np.errstate(divide='ignore'):  # ignore rare cases where 0 is encountered
        edc_db = 10 * np.log10(edc)
    edc_db[np.isneginf(edc_db)] = np.min(edc_db)
    return edc, edc_db


def get_rt_from_edc(edc_db, fs, offset_db=15, rt_interval_db=20):
    # normalize initial value of EDC top 0 dB
    edc_db -= edc_db[0]

    # time vector for EDC samples
    t = np.arange(len(edc_db)) / fs

    # fit line(y=ax + b) to EDC starting from -offset_db down to - offset_db - rt_interval_db
    # find sample indices of starting and ending point of the fit
    idx1 = np.argmin(np.abs(edc_db + offset_db))
    idx2 = np.argmin(np.abs(edc_db + (offset_db + rt_interval_db)))

    # compute line params
    y1 = edc_db[idx1]
    y2 = edc_db[idx2]
    x1 = t[idx1]
    x2 = t[idx2]
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    rt60 = -60 / a

    '''t2 = np.arange(rt60 * fs) / fs
    plt.plot(t, edc_db)
    plt.plot(t2, a * t2 + b)
    plt.plot(t, np.ones(np.size(t)) * -60)
    plt.scatter(rt60, -60)
    plt.show()'''

    return rt60


if __name__ == '__main__':
    pass
