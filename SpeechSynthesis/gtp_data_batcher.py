import numpy as np
import random

class G2PDataBatcher:
    def __init__(self, file_path):
        self.word_character_map = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7, "i": 8, "j": 9, "k": 10, "l": 11, "m": 12, "n":13, "o": 14, "p": 15, "q": 16, "r": 17, "s": 18, "t": 19, "u": 20, "v": 21, "w": 22, "x": 23, "y": 24, "z": 25, ",": 26, "'": 27}
        self.phoneme_map = {}
        self.max_word_length = 0
        self.max_phoneme_length = 0
        self.sequence_length = 50
        word_phoneme_pairs = []
        phoneme_index = 0
        with open(file_path, "r") as data_file:
            for line in data_file:
                tokens = line.split("-")
                word = tokens[0]
                if not "0" in word and not "1" in word and not "2" in word and not "3" in word:
                    if len(word) > self.max_word_length:
                        self.max_word_length = len(word)
                    phonemes = tokens[1].split("_")
                    if len(phonemes) > self.max_phoneme_length:
                        self.max_phoneme_length = len(phonemes)
                    for p in phonemes:
                        p = p.strip()
                        if not p in self.phoneme_map:
                            self.phoneme_map[p] = phoneme_index
                            phoneme_index += 1
                    word_phoneme_pairs.append((word, phonemes))
        self.train_samples = []
        for word, phonemes in word_phoneme_pairs:
            word_tensor = np.zeros([self.sequence_length, len(self.word_character_map)], dtype=np.float32)
            for index, c in enumerate(word):
                char_index = self.word_character_map[c]
                word_tensor[index][char_index] = 1.0
            phoneme_tensor = np.zeros([self.sequence_length, len(self.phoneme_map)], dtype=np.float32)
            for index, p in enumerate(phonemes):
                p_index = self.phoneme_map[p.strip()]
                phoneme_tensor[index][p_index] = 1.0
            self.train_samples.append((word_tensor, phoneme_tensor))
        self.test_samples = [self.train_samples.pop(random.randrange(len(self.train_samples))) for _ in range(10000)]# self.train_samples[::10][:-1]
        # self.train_samples = [self.train_samples[i] for i in range(len(self.train_samples)) if (i+1) % 10 != 0]
        self.epoch_samples = list(self.train_samples)

    def epoch_finished(self):
        return len(self.epoch_samples) == 0

    def prepare_epoch(self):
        self.epoch_samples = list(self.train_samples)

    def get_training_batch(self, size):
        batch = [self.epoch_samples.pop(random.randrange(len(self.epoch_samples))) for _ in range(size)]
        return self.__prepare_batch(batch)

    def get_test_batches(self, batch_sizes):
        grapheme_batch, phoneme_batch = self.__prepare_batch(self.test_samples)
        return np.split(grapheme_batch, batch_sizes, axis=0), np.split(phoneme_batch, batch_sizes, axis=0)

    def __prepare_batch(self, batch):
        grapheme_batch = []
        phoneme_batch = []
        for graph, phon in batch:
            grapheme_batch.append(graph)
            phoneme_batch.append(phon)
        return np.array(grapheme_batch), np.array(phoneme_batch)
