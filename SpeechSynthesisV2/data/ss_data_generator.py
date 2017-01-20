import os
import soundfile as sf
import math
import numpy as np

def generate_all_data(phone_map_file_path, root_dir_path, output_dir_path, batch_size=50):
    pm, pmi = get_phone_maps(phone_map_file_path)
    data_dirs = find_data_dirs(root_dir_path)
    data_tuples = get_data_tuples(data_dirs)
    lps, ls = get_data_stats(data_tuples, 1024, 512)

    window = make_hanning_window(1024)
    sample_index = 0
    phone_batch = list()
    spectrogram_batch = list()
    for phones, sound_file_path in data_tuples:
        phones_tensor = np.zeros(lps)
        for i in range(len(phones)):
            phones_tensor[i] = pm[phones[i]]
        with open(sound_file_path, "rb") as sndfile:
            data, sample_rate = sf.read(sndfile)
            index = 0
            specs = list()
            fft_len = 0
            while index < len(data) - (1024 + 512):
                spec = np.fft.rfft(window * data[index:index+1024])
                spec = np.transpose(np.array([[spec.real, spec.imag]]), (0, 2, 1))
                fft_len = spec.shape[1]
                specs.append(spec)
                index += 512
            spectrogram = np.concatenate(specs, axis=0) # np.zeros([ls - len(specs), fft_len, 2]
            spectrogram = np.concatenate([spectrogram, np.zeros([ls - len(specs), fft_len, 2])], axis=0)

            # Actually save the pair
            phone_batch.append(phones_tensor)
            spectrogram_batch.append(spectrogram)
        if len(phone_batch) == batch_size:
            print("Saving sample {}".format(sample_index))
            batch_phones = np.concatenate(phone_batch, axis=0)
            batch_specs = np.concatenate(spectrogram_batch, axis=0)
            phone_batch = list()
            spectrogram_batch = list()
            np.savez_compressed(os.path.join(output_dir_path, "sample_{}".format(sample_index)), [batch_phones, batch_specs], ["phones", "spectrograms"])
            sample_index += 1


def get_data_stats(data_tuples, window_size, hop_size):
    longest_phone_seq = 0
    longest_spectrogram = 0
    for phones, sound_file_path in data_tuples:
        if len(phones) > longest_phone_seq:
            longest_phone_seq = len(phones)
        with open(sound_file_path, "rb") as sndfile:
            data, sample_rate = sf.read(sndfile)
            spect_len = len(data) // hop_size + 1
            if spect_len > longest_spectrogram:
                longest_spectrogram = spect_len
    return longest_phone_seq, longest_spectrogram

def get_data_tuples(data_dirs):
    data_tuples = list()
    for data_dir in data_dirs:
        for dirpath, dirnames, filenames in os.walk(data_dir):
            transcript_file_names = list(filter(lambda x: "phon_transcript" in x, filenames))
            for file_name in transcript_file_names:
                with open(os.path.join(dirpath, file_name), "r") as transcript_file:
                    for line in transcript_file:
                        tokens = line.split(" ")
                        sound_file_path = "{}.flac".format(os.path.join(dirpath, tokens[0]))
                        phones = list(map(lambda x: x.strip(), tokens[1:]))
                        data_tuples.append((phones, sound_file_path))
            break
    return data_tuples

def get_phone_maps(phone_map_file_path):
    phone_map = dict()
    with open(phone_map_file_path, "r") as phone_map_file:
        for line in phone_map_file:
            tokens = line.split(" ")
            phone_map[tokens[0]] = int(tokens[1])
    phone_map_inv = {index: phone for phone, index in phone_map.items()}
    return phone_map, phone_map_inv

def generate_phone_map(root_dir_path, output_file_path):
    dirs = find_data_dirs(root_dir_path)
    phone_set = discover_phones(dirs)
    with open(os.path.join(output_file_path), "w") as phone_map_file:
        for index, phone in enumerate(phone_set):
            phone_map_file.write("{} {}\n".format(phone, index+1))

def discover_phones(data_dirs):
    phone_set = set()
    for data_dir in data_dirs:
        for dirpath, dirnames, filenames in os.walk(data_dir):
            transcript_file_name = list(filter(lambda x: "phon_transcript" in x, filenames))[0]
            with open(os.path.join(dirpath, transcript_file_name), "r") as phon_file:
                for line in phon_file:
                    tokens = line.split(" ")
                    for phon in tokens[1:]:
                        phon = phon.strip()
                        if not phon in phone_set:
                            phone_set.add(phon)
            break
    return phone_set

def find_data_dirs(root_path):
    dirs = []
    find_data_dirs_rec(root_path, dirs)
    return dirs

def find_data_dirs_rec(root_path, dir_list):
    for dirpath, dirnames, filenames in os.walk(root_path):
        if len(filenames) > 0:
            flacs = list(filter(lambda x: x.endswith(".flac"), filenames))
            if len(flacs) > 0:
                dir_list.append(root_path)
        for dirname in dirnames:
            path = os.path.join(dirpath, dirname)
            find_data_dirs_rec(path, dir_list)
        break

def make_hanning_window(size):
    window = np.zeros(size)
    for i in range(size):
        window[i] = 0.5 * (1.0 - math.cos((2.0 * math.pi * i) / (size - 1)))
    return window

generate_all_data(os.path.join("generated", "phone_map.txt"), "text-phone-sound-test", os.path.join("generated", "test_samples"))
