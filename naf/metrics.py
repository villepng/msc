import numpy as np


def get_edc(rir, normalize=True):
    # compute the inverse cumulative sum of the squared RIR (that's the EDC)
    rir2_flipped = np.flip(rir ** 2)
    rir2_flipped_csum = np.zeros(np.size(rir))
    rir2_flipped_csum[0] = rir2_flipped[0]
    for n in range(1, len(rir)):
        rir2_flipped_csum[n] = rir2_flipped_csum[n - 1] + rir2_flipped[n]
    edc = np.flip(rir2_flipped_csum)
    if normalize:
        edc = edc/edc[0]
    edc_db = 10 * np.log10(edc)

    return edc, edc_db


def get_rt_from_edc(edc_db, fs, offset_db=5, rt_interval_db=30):
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

    return rt60


if __name__ == '__main__':
    pass
