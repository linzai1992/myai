import numpy as np

class VocabHandler:
	def __init__(self, embeddings_path):
		self.vocab_size = 400000
		self.dimensions = 50
		self.vocab_tensor = np.zeros((self.vocab_size, self.dimensions), dtype=np.float32)
		self.vocab_map = {}
		with open(embeddings_path, "r") as emb_file:
			word_index = 0
			for line in emb_file:
				tokens = line.split(" ")
				word = tokens[0]
				values = tokens[1:]
				self.vocab_map[word] = word_index
				for index, val in enumerate(values):
					self.vocab_tensor[word_index][index] = float(val.strip())

print("Loading word embeddings...")
v = VocabHandler("data/glove.6B.50d.txt")
