import json
import random
import numpy as np

class DataBatcher:
	def __init__(self, data_path):
		self.sos_token = "<S>"
		self.eos_token = "<E>"
		self.unk_token = "<U>"
		self.pad_token = "<P>"
		with open(data_path, "r") as json_file:
			raw_json = json.load(json_file)
			index = 4
			key_index = 0
			self.word_map = {self.sos_token: 3, self.eos_token: 1, self.unk_token: 2, self.pad_token: 0}
			self.key_map = {}
			self.max_sentence_len = 0
			for key, sentences in raw_json.items():
				if not key in self.key_map:
					self.key_map[key] = key_index
					key_index += 1
				for sentence in sentences:
					tokens = sentence.split(" ")
					if len(tokens) > self.max_sentence_len:
						self.max_sentence_len = len(tokens)
					for token in tokens:
						if not token in self.word_map:
							self.word_map[token] = index
							index += 1
			self.total_samples = 0
			for key in raw_json:
				self.total_samples += len(raw_json[key])
			self.max_sentence_len += 2
			self.index_map = dict([reversed(i) for i in self.word_map.items()])
			self.index_key_map = dict([reversed(i) for i in self.key_map.items()])
			self.total_words = len(self.word_map)
			self.total_classes = len(self.key_map)
			self.data = [(sent, label) for label, sentences in raw_json.items() for sent in sentences]
			self.epoch_data = list(self.data)

	def prepare_epoch(self):
		self.epoch_data = list(self.data)

	def epoch_finished(self):
		return len(self.epoch_data) == 0

	def generate_batch(self, size):
		if size > len(self.epoch_data):
			size = len(self.epoch_data)
		batch = [self.epoch_data.pop(random.randrange(len(self.epoch_data))) for _ in range(size)]
		sentence_tensors = []
		label_tensors = []
		for sentence, label in batch:
			tensor = self.sentence_to_tensor(sentence)
			sentence_tensors.append(tensor)
			label_tensor = np.zeros(self.total_classes)
			label_tensor[self.key_map[label]] = 1
			label_tensors.append(label_tensor)
		return np.array(sentence_tensors), np.array(label_tensors)

	def generate_full_batch(self):
		batch = list(self.data)
		sentence_tensors = []
		label_tensors = []
		for sentence, label in batch:
			tensor = self.sentence_to_tensor(sentence)
			sentence_tensors.append(tensor)
			label_tensor = np.zeros(self.total_classes)
			label_tensor[self.key_map[label]] = 1
			label_tensors.append(label_tensor)
		return np.array(sentence_tensors), np.array(label_tensors)

	def sentence_to_tensor(self, sentence):
		tokens = sentence.split(" ")
		tokens.insert(0, self.sos_token)
		tokens.append(self.eos_token)
		tensor = np.full((self.max_sentence_len, self.total_words, 1), self.word_map[self.pad_token], dtype=np.float32)
		for index, token in enumerate(tokens):
			tensor[index][self.word_map[token]][0] = 1
		return np.array(tensor)

	def preprocess_string(self, strg):
		strg = strg.lower().replace(".", "").replace("?", "").replace("!", "").replace(",", "").strip()
		tokens = strg.split(" ")
		final = ""
		for token in tokens:
			if token in self.word_map:
				final += token + " "
			else:
				final += self.unk_token + " "
		return final[:-1]







