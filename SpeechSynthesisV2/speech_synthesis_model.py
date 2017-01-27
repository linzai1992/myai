import tensorflow as tf

class SpeechSynthesisModel:
    def __init__(self, input_sequence_length, vocab_size, output_sequence_length, output_size):
        with tf.variable_scope("speech_synthesis_seq2seq"):
            with tf.variable_scope("inputs"):
                self.inputs = tf.placeholder(tf.int32, shape=[None, input_sequence_length])
                embedded_inputs = tf.squeeze(self.embedding_layer(vocab_size, 64, self.inputs), axis=-1)

            with tf.variable_scope("encoder"):
                encoder_rnn_cell = tf.nn.rnn_cell.GRUCell(256)
                encoder_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([encoder_rnn_cell] * 2)
                encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_rnn_cell, embedded_inputs, dtype=tf.float32)

            with tf.variable_scope("context"):
                encoder_output_last = tf.transpose(encoder_output, perm=[0, 2, 1])
                context_weights = tf.Variable(tf.truncated_normal([1, input_sequence_length, output_sequence_length], stddev=0.1))
                context_tensor = tf.transpose(tf.batch_matmul(encoder_output_last, context_weights), perm=[0, 2, 1])

            with tf.variable_scope("decoder"):
                decoder_rnn_cell = tf.nn.rnn_cell.GRUCell(242)
                decoder_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([decoder_rnn_cell] * 2)
                decoder_output, decoder_state = tf.nn.dynamic_rnn(decoder_rnn_cell, context_tensor, dtype=tf.float32)

                print(decoder_output.get_shape())

    def embedding_layer(self, vocab_size, embedding_size, input_tensor):
        with tf.device("/cpu:0"):
            emb_weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=0.1))
            embedding_raw = tf.nn.embedding_lookup(emb_weights, input_tensor)
            embedding = tf.expand_dims(embedding_raw, -1)
            return embedding

m = SpeechSynthesisModel(256, 98, 512, 512)
