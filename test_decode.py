import numpy as np
import pathlib
import spaudiopy as spa  # currently requires python versions >=3.6 but < 3.12

from scipy.io import wavfile


# todo: all test files in loop
def main():
    parent_dir = str(pathlib.Path.cwd().parent)
    audio_data_path = f'{parent_dir}/data/generated/test/testset/subject1'
    order = 1
    components = (order + 1) ** 2
    fs, mono = wavfile.read(f'{audio_data_path}/mono.wav')

    ambisonic = np.zeros((components, len(mono)))
    for i in range(components):
        _, part = wavfile.read(f'{audio_data_path}/ambisonic_{i}.wav')
        ambisonic[i, :] = part
    hrir, _ = spa.io.sofa_to_sh('C:/Users/Ville/Documents/VS/data/irs etc/mit_kemar_normal_pinna.sofa', order)

    binaural = spa.decoder.sh2bin(ambisonic, hrir)
    wavfile.write(f'{parent_dir}/data/out/binaural_1.wav', fs, binaural.astype(np.int16).T)


if __name__ == '__main__':
    main()
