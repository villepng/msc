import numpy as np
import pathlib
import spaudiopy as spa  # currently requires python versions >=3.6 but < 3.12

from scipy.io import wavfile


def main():
    parent_dir = str(pathlib.Path.cwd().parent)
    order = 2
    components = (order + 1) ** 2

    for i in range(1, 13):
        audio_data_path = f'{parent_dir}/data/generated/rir_ambisonic_test/testset/subject{i}'
        fs, mono = wavfile.read(f'{audio_data_path}/mono.wav')

        ambisonic = np.zeros((components, len(mono)))
        for j in range(components):
            _, part = wavfile.read(f'{audio_data_path}/ambisonic_{j}.wav')
            ambisonic[j, :] = part
        hrir, _ = spa.io.sofa_to_sh('C:/Users/Ville/Documents/VS/data/irs etc/mit_kemar_normal_pinna.sofa', order)

        binaural = spa.decoder.sh2bin(ambisonic, hrir)
        wavfile.write(f'{parent_dir}/data/out/binaural_{i}.wav', fs, binaural.astype(np.int16).T)


if __name__ == '__main__':
    main()
