import matplotlib.pyplot as plt
import numpy as np

from masp import shoebox_room_sim as srs


room_xyz = np.array([10.0, 6.0, 2.5])
rec_xyz = np.array([[9.0, 5.0, 1.5]])
src_xyz = np.array([[1.0, 1.0, 1.5]])
mic_specs = np.array([[1, 0, 0, 1]])
band_centerfreqs = np.array([125, 250, 500, 1000, 2000, 4000])
abs_wall = np.array([[0.04, 0.04, 0.03, 0.03, 0.02, 0.02],
                    [0.04, 0.04, 0.03, 0.03, 0.02, 0.02],
                    [0.06, 0.2, 0.65, 0.9, 0.95, 0.98],
                    [0.06, 0.2, 0.65, 0.9, 0.95, 0.98],
                    [0.08, 0.08, 0.3, 0.6, 0.75, 0.8],
                    [0.2, 0.15, 0.1, 0.08, 0.04, 0.02]])
nBands = abs_wall.shape[0]
band_centerfreqs = np.empty(nBands)
band_centerfreqs[0] = 125
for nb in range(1, nBands):
    band_centerfreqs[nb] = 2 * band_centerfreqs[nb-1]
limits = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5])
fs = 16000
abs_echograms = srs.compute_echograms_mic(room_xyz, src_xyz, rec_xyz, abs_wall, limits, mic_specs)
mic_rirs = srs.render_rirs_mic(abs_echograms, band_centerfreqs, fs).squeeze()

samples = int(1.4 * 16000)
t = np.arange(samples) / fs
plt.plot(t, mic_rirs[:samples] * mic_rirs[:samples])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

