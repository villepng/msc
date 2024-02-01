import matplotlib.pyplot as plt
import numpy as np
import scipy

from naf.data_loading.data_maker import GetSpec


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
    # compute the inverse cumulative sum of the squared RIR (that's the EDC)
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


def get_rt_from_edc(edc_db, fs, offset_db=5, rt_interval_db=55):
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
    rt60 = (-60 - b) / a

    # plt.plot(t, edc_db)
    # plt.plot(t, a * t + b)
    # plt.plot(t, np.ones(np.size(t)) * -60)
    # plt.scatter(rt60, -60)
    # plt.show()

    # rt60 = t[np.argmin(np.abs(edc_db + 60))], this gives lower error?

    return rt60


if __name__ == '__main__':
    pass
