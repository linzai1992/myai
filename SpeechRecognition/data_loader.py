import soundfile as sf
from os import walk
import numpy as np
import random

class DataLoader:
	def __init__(self, root_data_path):
		# "/Users/tateallen/Documents/MyAI/SpeechRecognition/sound_data/dev-clean"
		self.msl, _, self.mul, _ = get_data_stats(root_data_path)
		self.all_data = extract_all_data(root_data_path, self.msl, self.mul)
		self.word_map = create_word_map(self.all_data)

	def get_batch(self, size):
		sample = random.sample(self.all_data, size)
		audio = []
		words = []
		for audio_data, tokens in sample:
			words.append(list(map(lambda x: self.word_map[x], tokens)))
			audio.append(audio_data)
		return np.array(audio).reshape(size, 1, self.mul), np.array(words).reshape(size, 1, self.msl)

def create_word_map(data):
	map = {}
	index = 0
	for audio_data, tokens in data:
		for token in tokens:
			if not token in map:
				map[token] = index
				index += 1
	return map

def extract_all_data(root_path, msl, mul):
	dirs = find_data_dirs(root_path)
	all_data = []
	for dir in dirs:
		data = extract_data(dir, msl, mul)
		all_data += data
	return all_data

def get_data_stats(root_path):
	dirs = find_data_dirs(root_path)
	max_sentence_length = 0
	average_sentence_length = 0
	max_snd_length = 0
	average_snd_length = 0
	for dir in dirs:
		msl, asl, mnl, anl = data_dir_info(dir)
		if msl > max_sentence_length:
			max_sentence_length = msl
		if mnl > max_snd_length:
			max_snd_length = mnl
		average_sentence_length += asl
		average_snd_length += anl
	average_sentence_length /= len(dirs)
	average_snd_length /= len(dirs)
	return max_sentence_length, average_sentence_length, max_snd_length, average_snd_length

def find_data_dirs(root_path):
	dirs = []
	find_data_dirs_rec(root_path, dirs)
	return dirs

def find_data_dirs_rec(root_path, dir_list):
	for (dirpath, dirnames, filenames) in walk(root_path):
		if len(filenames) > 0:
			flacs = list(filter(lambda x: x.endswith(".flac"), filenames))
			if len(flacs) > 0:
				dir_list.append(root_path)
		for dirname in dirnames:
			path = dirpath + "/" + dirname
			find_data_dirs_rec(path, dir_list)
		break

def extract_data(path, msl, mul):
	print("Extracting data from %s" % path)
	pair_data = []
	for (dirpath, dirnames, filenames) in walk(path):
		flacs = list(filter(lambda x: x.endswith(".flac"), filenames))
		temp_dict = {}
		for f in flacs:
			flac_path = dirpath + "/" + f
			with open(flac_path, "rb") as file:
				data, sample_rate = sf.read(file)
				modified_data = list(map(lambda x: int((((x - -1) * (255 - 0)) / (1 - -1)) + 0), data))
				padding_len = mul - len(modified_data)
				if padding_len > 0:
					modified_data += [0] * padding_len
				temp_dict[f] = np.array(modified_data) # + zeros
		txt = list(filter(lambda x: x.endswith(".txt"), filenames))[0]
		with open(dirpath + "/" + txt, "r") as file:
			for line in file:
				tokens = line.split(" ")
				id = tokens[0]
				tokens_2 = ["<BOS>"]
				tokens_2 += list(map(lambda x: x.lower().strip(), tokens[1:]))
				tokens_2 += ["<EOS>"]
				padding_len = msl - len(tokens_2)
				if padding_len > 0:
					tokens_2 += ["<PAD>"] * padding_len
				pair = (temp_dict[id+".flac"], tokens_2)
				pair_data.append(pair)
		break
	return pair_data

def data_dir_info(path):
	text_path = ""
	flacs = []
	for (dirpath, dirnames, filenames) in walk(path):
		txt = list(filter(lambda x: x.endswith(".txt"), filenames))[0]
		text_path = dirpath + "/" + txt
		flacs = list(filter(lambda x: x.endswith(".flac"), filenames))
		flacs = list(map(lambda x: dirpath + "/" + x, flacs))
		break

	max_sentence_length = 0
	average_sentence_length = 0
	with open(text_path, "r") as file:
		file_index = 0
		for line in file:
			tokens = line.split(" ")
			length = len(tokens) - 1
			if length > max_sentence_length:
				max_sentence_length = length
			average_sentence_length += length
			file_index += 1
		average_sentence_length /= file_index

	max_snd_length = 0
	average_snd_length = 0
	for sound_file in flacs:
		with open(sound_file, "rb") as sndfile:
			data, sample_rate = sf.read(sndfile)
			snd_length = len(data)
			if snd_length > max_snd_length:
				max_snd_length = snd_length
			average_snd_length += snd_length
	average_snd_length /= len(flacs)
	return max_sentence_length, average_sentence_length, max_snd_length, average_snd_length


dl = DataLoader("/Users/tateallen/Documents/MyAI/SpeechRecognition/sound_data/dev-clean-copy")
a, w = dl.get_batch(20)
print(a.shape)
print(a)
print(w)

