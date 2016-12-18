import tensorflow as tf

class SpeechRecognitionModel:
	def __init__(self, audio_length, sentence_length):
		self.encoder_inputs = tf.placeholder(tf.float32, shape=(None, 1, audio_length))
		self.decoder_inputs = tf.placeholder(tf.float32, shape=(None, 1, sentence_length))

		print("Created placeholders")
		rnn_cell = tf.nn.rnn_cell.GRUCell(256)
		network = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * 3)
		print("Created rnn cells")

		input = tf.unstack(self.encoder_inputs, axis=0, num=audio_length)
		print("Unstacked!")
		dec_input = tf.unstack(self.decoder_inputs, axis=0, num=sentence_length)
		print("Unstacked!")
		outputs, state = tf.nn.seq2seq.basic_rnn_seq2seq(input, dec_input, network)
		print("Created seq2seq")
