import matplotlib.pyplot as plt
import numpy as np

from masp import shoebox_room_sim as srs
from naf.data_loading.data_maker import GetSpec
from scipy.io import wavfile
from scipy.signal import fftconvolve

MATERIALS = {
    'carpet': [0.08, 0.08, 0.3, 0.6, 0.75, 0.8],  # Frequency bands; 125, 250, 500, 1k, 2k, 4k
    'concrete': [0.36, 0.44, 0.31, 0.29, 0.39, 0.25],
    'drapery': [0.14, 0.35, 0.53, 0.75, 0.7, 0.6],
    'fiberglass': [0.06, 0.2, 0.65, 0.9, 0.95, 0.98],
    'glass': [0.04, 0.04, 0.03, 0.03, 0.02, 0.02],
    'plaster': [0.2, 0.15, 0.1, 0.08, 0.04, 0.02],
    }

wall = [0.06, 0.2, 0.65, 0.9, 0.95, 0.98]
ceiling = [0.08, 0.08, 0.3, 0.6, 0.75, 0.8]
test = [0.01, 0.01, 0.99, 0.99, 0.01, 0.01]
# wall, ceiling = np.flip(wall), np.flip(ceiling)

room_xyz = np.array([10.0, 6.0, 2.5])
rec_xyz = np.array([[1.0, 5.0, 1.5]])
src_xyz = np.array([[1.0, 1.0, 1.5]])
mic_specs = np.array([[1, 0, 0, 1]])
band_centerfreqs = np.array([125, 250, 500, 1000, 2000, 4000])
abs_wall = np.array([[0.05,	0.12, 0.35,	0.48, 0.38, 0.36],   # drapery
                     [0.1, 0.07, 0.05, 0.04, 0.04, 0.04],    # glass
                     [0.4, 0.65, 0.85, 0.75, 0.65, 0.6],     # open brick pattern
                     [0.4, 0.65, 0.85, 0.75, 0.65, 0.6],     # open brick patter
                     [0.01, 0.02, 0.06, 0.15, 0.25, 0.45],   # carpet
                     [0.15, 0.11, 0.04, 0.04, 0.07, 0.08]]).T  # plasterboard
# abs_wall = np.array([wall, wall, wall, wall, ceiling, ceiling]).T
abs_wall = np.array([MATERIALS['glass'], MATERIALS['drapery'], MATERIALS['concrete'],
                     MATERIALS['fiberglass'], MATERIALS['carpet'], MATERIALS['plaster']]).T

if True and 0:  # basic absorption
    nBands = 1
    band_centerfreqs = np.empty(nBands)
    band_centerfreqs[0] = 1000
    rt60 = np.array([0.2])
    abs_wall = srs.find_abs_coeffs_from_rt(room_xyz, rt60)[0]
    abs_wall = np.array([test])
    maxlim = 0.8  # 0.8, 1s
    limits = np.empty(nBands)
    limits.fill(np.minimum(0.5, maxlim))
else:
    nBands = abs_wall.shape[0]
    # limits = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5])
    limits = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

fs = 16000
order = 0
# abs_echograms = srs.compute_echograms_mic(room_xyz, src_xyz, rec_xyz, abs_wall, limits, mic_specs)
# mic_rirs = srs.render_rirs_mic(abs_echograms, band_centerfreqs, fs).squeeze()

abs_echograms = srs.compute_echograms_sh(room_xyz, src_xyz, rec_xyz, abs_wall, limits, order)
sh_rirs = srs.render_rirs_sh(abs_echograms, band_centerfreqs, fs).squeeze()

plt.rcParams.update({'font.size': 22})
if order:
    t = np.arange(len(sh_rirs[:, 0])) / fs
    channels = ['W', 'Y', 'Z', 'X']
    fig, axarr = plt.subplots(2, 2)
    for channel, subfig in enumerate(axarr.flat):
        subfig.set_title(f'{channels[channel]} channel')
        subfig.set_xlim([0, 1.4])
        subfig.set_ylim([np.min(sh_rirs) * 1.1, np.max(sh_rirs) * 1.1])
        subfig.plot(t, sh_rirs[:, channel])
        subfig.set_ylabel('Amplitude')
        subfig.set_xlabel('Time (s)')
    plt.show()
else:
    t = np.arange(len(sh_rirs)) / fs
    plt.plot(t, sh_rirs)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

    _, audio_anechoic = wavfile.read('../data/timit/data/TRAIN/DR1/FCJF0/SA1.WAV.wav')
    reverberant_signal = fftconvolve(audio_anechoic, sh_rirs)
    reverberant_signal = reverberant_signal / max(reverberant_signal)
    wavfile.write(f'normal.wav', fs, reverberant_signal.astype(np.float32))

    spec_getter = GetSpec(components=1)
    sh_rirs = sh_rirs.reshape(1, len(sh_rirs))
    real_spec, _, _ = spec_getter.transform(sh_rirs)
    # plt.specgram(sh_rirs[0], 128, fs, noverlap=64, mode='magnitude', scale='dB')
    t = np.arange(real_spec.shape[-1]) / fs * 64
    f = np.arange(real_spec.shape[-2]) / (128 / fs)
    plt.pcolormesh(t, f, real_spec[0])
    # plt.yscale('log')
    plt.show()
