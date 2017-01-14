import os
from os import walk
import soundfile as sf
import numpy as np

class DataGenerator:
    def __init__(self):
        pass

    def generate_windows(self, input_dir_path, output_dir_path, window_size, overlapping=True):
        data_dirs = self.__find_data_dirs(input_dir_path)
        index = 0
        for data_dir in data_dirs:
            for (dirpath, dirnames, filenames) in walk(data_dir):
                flacs = list(filter(lambda x: x.endswith(".flac"), filenames))
                flacs = list(map(lambda x: dirpath + "/" + x, flacs))
                sub_tensors = []
                for flac_file_path in flacs:
                    windows_tensor = self.__generate_windows_tensor(flac_file_path, window_size, overlapping=overlapping)
                    sub_tensors.append(windows_tensor)
                giant_tensor = np.concatenate(sub_tensors, axis=0)
                save_path = os.path.join(output_dir_path, "windows_{}.npy".format(index))
                np.savez_compressed(save_path, [giant_tensor])
                index += 1
                print("Generated windows tensor %i" % (index-1), giant_tensor.shape)
                break

    def __generate_windows_tensor(self, flac_path, window_size, overlapping=True):
        with open(flac_path, "rb") as sndfile:
            data, sample_rate = sf.read(sndfile)
            step = overlapping if overlapping else window_size
            index = 0
            tensor = []
            while index + window_size < len(data):
                arr = np.array(data[index:index+window_size])
                tensor.append(arr)
                index += step
            return np.array(tensor)


    def __find_data_dirs(self, root_path):
    	dirs = []
    	self.__find_data_dirs_rec(root_path, dirs)
    	return dirs

    def __find_data_dirs_rec(self, root_path, dir_list):
    	for (dirpath, dirnames, filenames) in walk(root_path):
    		if len(filenames) > 0:
    			flacs = list(filter(lambda x: x.endswith(".flac"), filenames))
    			if len(flacs) > 0:
    				dir_list.append(root_path)
    		for dirname in dirnames:
    			path = os.path.join(dirpath, dirname)
    			self.__find_data_dirs_rec(path, dir_list)
    		break

gen = DataGenerator()
gen.generated_data("sound_data", "generated_data", 50, overlapping=False)
