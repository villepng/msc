import gc
import h5py
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import pickle
import torch
import torchaudio
import tqdm

from scipy.interpolate import interp1d
from scipy.io import wavfile
from skimage.transform import rescale, resize
from torchaudio.transforms import Spectrogram

from naf.options import Options
from naf.test_query import to_wave_if


class GetSpec:
    def __init__(self, sr=16000, use_torch=False, power_mod=2, fft_size=512):
        self.sr = sr
        self.n_fft = fft_size
        self.hop = self.n_fft // 4
        if use_torch:
            assert False
            self.use_torch = True
            self.spec_transform = Spectrogram(power=None, n_fft=self.n_fft, hop_length=self.hop)
        else:
            self.power = power_mod
            self.use_torch = False
            self.spec_transform = None

    def transform(self, wav_data_prepad):
        wav_data = librosa.util.fix_length(wav_data_prepad, size=wav_data_prepad.shape[-1] + self.n_fft // 2)
        if wav_data.shape[1] < self.sr * 0.1:
            wav_data = librosa.util.fix_length(wav_data, size=int(self.sr * 0.1))
        if self.use_torch:
            transformed_data = self.spec_transform(torch.from_numpy(wav_data)).numpy()
        else:
            transformed_data = np.array([librosa.stft(wav_data[0], n_fft=self.n_fft, hop_length=self.hop)])[:, :-1]  # mono
            # transformed_data = np.array([librosa.stft(wav_data[0], n_fft=self.n_fft, hop_length=self.hop),
            #                             librosa.stft(wav_data[1], n_fft=self.n_fft, hop_length=self.hop)])[:, :-1]
        #         print(np.array([librosa.stft(wav_data[0],n_fft=self.n_fft, hop_length=self.hop),
        #                librosa.stft(wav_data[1],n_fft=self.n_fft, hop_length=self.hop)]).shape, 'OLD SHAPE')

        real_component = np.abs(transformed_data)
        img_component = np.angle(transformed_data)
        gen_if = if_compute(img_component) / np.pi
        return np.log(real_component + 1e-3), gen_if, img_component


def if_compute(arg):
    unwrapped_angle = np.unwrap(arg).astype(np.single)
    return np.concatenate([unwrapped_angle[:, :, 0:1], np.diff(unwrapped_angle, n=1)], axis=-1)


# Misc. functions to make spectrograms
def load_audio(path_name, use_torch=True, resample=True, resample_rate=22050):
    # returns in shape (ch, num_sample), as float32 (on Linux at least)
    # by default torchaudio is wav_arr, sample_rate
    # by default wavfile is sample_rate, wav_arr
    if use_torch:
        loaded = torchaudio.load(path_name)
        wave_data_loaded = loaded[0].numpy()
        sr_loaded = loaded[1]
    else:
        loaded = wavfile.read(path_name)
        wave_data_loaded = np.clip(loaded[1], -1.0, 1.0).T
        sr_loaded = loaded[0]

    if resample:
        if wave_data_loaded.shape[1] == 0:
            print('len 0')
            assert False
        if wave_data_loaded.shape[1] < int(sr_loaded * 0.1):
            padded_wav = librosa.util.fix_length(wave_data_loaded, int(sr_loaded * 0.1))
            resampled_wave = librosa.resample(padded_wav, orig_sr=sr_loaded, target_sr=resample_rate)
        else:
            resampled_wave = librosa.resample(wave_data_loaded, orig_sr=sr_loaded, target_sr=resample_rate)
    else:
        resampled_wave = wave_data_loaded
    return np.clip(resampled_wave, -1.0, 1.0)


def pad(input_arr, max_len_in, constant=np.log(1e-3)):
    return np.pad(input_arr, [[0, 0], [0, 0], [0, max_len_in - input_arr.shape[2]]], constant_values=constant)


def resample(wave_data, sr=16000, resample_rate=22050):
    if wave_data.shape[1] == 0:
        print('len 0')
        assert False
    if wave_data.shape[1] < int(sr * 0.1):
        padded_wav = librosa.util.fix_length(wave_data, size=int(sr * 0.1))
        resampled_wave = librosa.resample(padded_wav, orig_sr=sr, target_sr=resample_rate)
    else:
        resampled_wave = librosa.resample(wave_data, orig_sr=sr, target_sr=resample_rate)
    return np.clip(resampled_wave, -1.0, 1.0)


# todo: a lot of cleanup
def main():
    args = Options().parse()
    base_path = f'../metadata/{args.type}{args.grid}'  # todo: needs order for ambisonics, should update general file structure
    mag_path = f'{base_path}/magnitudes'
    pathlib.Path(mag_path).mkdir(parents=True, exist_ok=True)
    phase_path = f'{base_path }/phases'
    pathlib.Path(phase_path).mkdir(parents=True, exist_ok=True)
    rooms = ['test_1']
    max_len_dict = {}
    spec_getter = GetSpec()
    with open(f'../../../data/generated/rirs/order_{args.order}/room_10.0x6.0x2.5/grid_{args.grid}/rt60_0.2/rirs.pickle', 'rb') as f:
        rirs = pickle.load(f)

    for room_name in rooms:
        length_tracker = []
        f_mag = h5py.File(f'{mag_path}/{room_name}.h5', 'w')
        f_phase = h5py.File(f'{phase_path}/{room_name}.h5', 'w')
        for orientation in ['0']:  # , '90', '180', '270']: might not need these
            progress = tqdm.tqdm(rirs.items())
            progress.set_description('Calculating spectrograms')
            for coordinate, rir in progress:
                # resampled = resample(np.clip(rir, -1.0, 1.0).T)
                resampled = rir.T
                real_spec, img_spec, raw_phase = spec_getter.transform(resampled)
                length_tracker.append(real_spec.shape[2])

                # sr = 16000
                # reconstructed_wave = to_wave_if(real_spec, img_spec)
                # fig, axes = plt.subplots(2, 1)
                # axes[0].plot(np.arange(len(reconstructed_wave)) / sr, reconstructed_wave)  # sr depends on resampling
                # axes[0].set_title('Reconstructed waveform')
                # axes[0].set_xlim([0, 0.2])
                # axes[1].plot(np.arange(len(rir)) / sr, rir)
                # axes[1].set_title('Original waveform')
                # axes[1].set_xlim([0, 0.2])
                # plt.show()
                # plt.imshow(real_spec[0])
                # plt.show()
                # plt.imshow(img_spec[0])
                # plt.show()

                f_mag.create_dataset('{}_{}'.format(orientation, coordinate.replace('-', '_')), data=real_spec.astype(np.half))
                f_phase.create_dataset('{}_{}'.format(orientation, coordinate.replace('-', '_')), data=img_spec.astype(np.half))
        print('Max length {}'.format(room_name), np.max(length_tracker))
        max_len_dict.update({room_name: np.max(length_tracker)})
        f_mag.close()
        f_phase.close()

    raw_path = mag_path
    mean_std = f'{base_path}/mean_std'
    pathlib.Path(mean_std).mkdir(parents=True, exist_ok=True)
    for f_name_old in sorted(list(max_len_dict.keys())):
        f_name = f'{f_name_old}.h5'
        print(f'Processing {f_name}')
        f = h5py.File(f'{raw_path}/{f_name}', 'r')
        keys = list(f.keys())
        max_len = max_len_dict[f_name.split('.')[0]]
        all_arrs = []
        # for idx in np.random.choice(len(keys), 4000, replace=False):
        #    all_arrs.append(pad(f[keys[idx]], max_len).astype(np.single))
        for key in keys:
            all_arrs.append(pad(f[key], max_len).astype(np.single))
        print('Computing mean')
        mean_val = np.mean(all_arrs, axis=(0, 1))
        print('Computing std')
        std_val = np.std(all_arrs, axis=(0, 1)) + 0.1

        plt.imshow(all_arrs[0][0])
        plt.show()
        plt.imshow(mean_val)
        plt.show()
        plt.imshow(std_val)
        plt.show()
        print(mean_val.shape)
        del all_arrs
        f.close()
        gc.collect()
        with open(f'{mean_std}/{f_name.replace("h5", "pkl")}', 'wb') as mean_std_file:
            pickle.dump([mean_val, std_val], mean_std_file)


if __name__ == '__main__':
    main()
