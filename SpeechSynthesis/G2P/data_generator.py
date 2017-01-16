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
        print("Building model...")
        self.model = G2PModel(self.batcher.sequence_length, len(self.batcher.word_character_map), len(self.batcher.phoneme_map))
        self.saver = tf.train.Saver()
        self.session = tf.Session()
        print("Loading model checkpoint...")
        ckpt = tf.train.get_checkpoint_state(model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print("Failed loading model!")

generator = DataGenerator("checkpoints")
