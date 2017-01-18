import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from g2p_model import G2PModel
from g2p_data_batcher import G2PDataBatcher
import numpy as np
import tensorflow as tf

class DataGenerator:
    def __init__(self, model_checkpoint_path):
        print("Loading data...")
        self.batcher = G2PDataBatcher("data/cmudict_proc.txt")

    def generate_data(self, sound_data_path):
        with tf.Session() as session:
            print("Building model...")
            self.model = G2PModel(self.batcher.sequence_length, len(self.batcher.word_character_map), len(self.batcher.phoneme_map))
            self.saver = tf.train.Saver()
            print("Loading model checkpoint...")
            ckpt = tf.train.get_checkpoint_state(model_checkpoint_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(session, ckpt.model_checkpoint_path)
            else:
                print("Failed loading model!")

            print("Getting metadata")
            meta_data = self.__get_meta_data(sound_data_path)
            print("Done size {}".format(len(m)))
            for d in meta_data:
                # 1. Generate phoneme list -> pass thru model
                # 2. Convert phoneme list to int32 list
                # 3. Convert sound data to windowed tensor
                # 4. Save them somehow
                

    def __get_meta_data(self, sound_data_path):
        dirs = self.__find_data_dirs(sound_data_path)
        data_points = []
        for d in dirs:
            for dirpath, dirnames, filenames in os.walk(d):
                transcript = list(filter(lambda x: x.endswith(".trans.txt"), filenames))[0]
                with open(os.path.join(dirpath, transcript)) as trans:
                    for line in trans:
                        tokens = line.split(" ")
                        sound_id = tokens[0]
                        words = tokens[1:]
                        if len(words) <= 9:
                            data_point = (sound_id, tokens, os.path.join(dirpath, "{}.flac".format(sound_id)))
                            data_points.append(data_point)
                break
        return data_points

    def __find_data_dirs(self, root_path):
    	dirs = []
    	self.__find_data_dirs_rec(root_path, dirs)
    	return dirs

    def __find_data_dirs_rec(self, root_path, dir_list):
    	for (dirpath, dirnames, filenames) in os.walk(root_path):
    		if len(filenames) > 0:
    			flacs = list(filter(lambda x: x.endswith(".flac"), filenames))
    			if len(flacs) > 0:
    				dir_list.append(root_path)
    		for dirname in dirnames:
    			path = os.path.join(dirpath, dirname)
    			self.__find_data_dirs_rec(path, dir_list)
    		break

generator = DataGenerator("checkpoints")
generator.generate_data("../VAE/sound_data/dev-clean")
