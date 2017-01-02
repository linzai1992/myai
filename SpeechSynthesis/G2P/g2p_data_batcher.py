import numpy as np
import random

class G2PDataBatcher:
    def __init__(self, file_path):
        alphabet = "abcdefghijklmnopqrstuvwxyz,'"
        self.word_character_map = {alphabet[i]: i+1 for i in range(len(alphabet))}
        self.word_character_map.update({"<PAD>": 0})
        self.phoneme_map = {"<PAD>": 0, "<GO>": 1, "<END>": 2}
        self.max_word_length = 0
        self.max_phoneme_length = 0
        self.sequence_length = 40
        word_phoneme_pairs = []
        phoneme_index = 3
        with open(file_path, "r") as data_file:
            for line in data_file:
                tokens = line.split("-")
                word = tokens[0]
                if len(word) > self.max_word_length:
                    self.max_word_length = len(word)
                phonemes = tokens[1].split("_")
                phonemes.append("<END>")
                phonemes = list(map(lambda x: x.strip(), phonemes))
                if len(phonemes) > self.max_phoneme_length:
                    self.max_phoneme_length = len(phonemes)
                for p in phonemes:
                    if not p in self.phoneme_map:
                        self.phoneme_map[p] = phoneme_index
                        phoneme_index += 1
                word_phoneme_pairs.append((word, phonemes))

        self.phoneme_map_inverse = {v: k for k, v in self.phoneme_map.items()}
        self.train_samples = []
        for word, phonemes in word_phoneme_pairs:
            word_tensor = np.full([self.sequence_length], self.word_character_map["<PAD>"], dtype=np.int32)
            for i in range(len(word)):
                word_tensor[i] = self.word_character_map[word[i]]
            phoneme_tensor = np.full([self.sequence_length], self.phoneme_map["<PAD>"], dtype=np.int32)
            for i in range(len(phonemes)):
                phoneme_tensor[i] = self.phoneme_map[phonemes[i]]
            self.train_samples.append((word_tensor, phoneme_tensor))
        self.test_samples = [self.train_samples.pop(random.randrange(len(self.train_samples))) for _ in range(2000)]# self.train_samples[::10][:-1]
        self.epoch_samples = list(self.train_samples)

    def epoch_finished(self):
        return len(self.epoch_samples) == 0

    def prepare_epoch(self):
        self.epoch_samples = list(self.train_samples)

    def get_training_batch(self, size):
        if size > len(self.epoch_samples):
            size = len(epoch_samples)
        batch = [self.epoch_samples.pop(random.randrange(len(self.epoch_samples))) for _ in range(size)]
        return self.__prepare_batch(batch)

    def get_test_batch(self):
        return self.__prepare_batch(self.test_samples)

    def batch_input_string(self, input_string):
        words = input_string.split(" ")
        batch = np.full([len(words), self.sequence_length], self.word_character_map["<PAD>"], dtype=np.int32)
        for i, word in enumerate(words):
            for j, c in enumerate(word):
                batch[i][j] = self.word_character_map[c]
        grapheme_batch = [b.reshape([len(words)]) for b in np.split(batch, self.sequence_length, axis=1)]
        return grapheme_batch

    def __prepare_batch(self, batch):
        grapheme_batch = []
        phoneme_batch = []
        for graph, phon in batch:
            grapheme_batch.append(graph)
            phoneme_batch.append(phon)
        grapheme_batch = np.array(grapheme_batch)
        phoneme_batch = np.array(phoneme_batch)
        grapheme_batch = [b.reshape([len(batch)]) for b in np.split(grapheme_batch, self.sequence_length, axis=1)]
        phoneme_batch = [b.reshape([len(batch)]) for b in np.split(phoneme_batch, self.sequence_length, axis=1)]
        return grapheme_batch, phoneme_batch
