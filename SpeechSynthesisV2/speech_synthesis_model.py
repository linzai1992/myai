import tensorflow as tf

class SpeechSynthesisModel:
    def __init__(self, input_sequence_length, vocab_size, output_sequence_length, output_size):
        with tf.variable_scope("speech_synthesis_seq2seq"):
            with tf.variable_scope("placeholders"):
                self.inputs = tf.placeholder(tf.int32, shape=[None, input_sequence_length])
                self.labels = tf.placeholder(tf.float32, shape=[None, output_sequence_length, output_size, 2])

                


            # with tf.variable_scope("placeholders"):
            #     self.inputs = tf.placeholder(tf.int32, shape=[None, input_sequence_length])
            #     self.labels = tf.placeholder(tf.float32, shape=[None, output_sequence_length, output_size, 2])
            #
            # with tf.variable_scope("inputs"):
            #     embedded_inputs = tf.squeeze(self.embedding_layer(vocab_size, 64, self.inputs), axis=-1)
            #
            # with tf.variable_scope("encoder"):
            #     encoder_rnn_cell = tf.contrib.rnn.core_rnn_cell.GRUCell(output_size)
            #     encoder_rnn_cell = tf.contrib.rnn.core_rnn_cell.MultiRNNCell([encoder_rnn_cell] * 2)
            #     encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_rnn_cell, embedded_inputs, dtype=tf.float32)
            #
            # # with tf.variable_scope("context"):
            # #     encoder_output_last = tf.transpose(encoder_output, perm=[0, 2, 1])
            # #     context_weights = tf.Variable(tf.truncated_normal([1, input_sequence_length, output_sequence_length], stddev=0.1))
            # #     context_tensor = tf.transpose(tf.batch_matmul(encoder_output_last, context_weights), perm=[0, 2, 1])
            #
            # with tf.variable_scope("decoder_mag"):
            #     decoder_rnn_cell_mag = tf.contrib.rnn.core_rnn_cell.GRUCell(output_size)
            #     decoder_rnn_cell_mag = tf.contrib.rnn.core_rnn_cell.MultiRNNCell([decoder_rnn_cell_mag] * 2)
            #     decoder_output_mag, decoder_state_mag = tf.nn.dynamic_rnn(decoder_rnn_cell_mag, encoder_output, dtype=tf.float32, initial_state=encoder_state)
            #
            # with tf.variable_scope("decoder_phz"):
            #     decoder_rnn_cell_phz = tf.contrib.rnn.core_rnn_cell.GRUCell(output_size)
            #     decoder_rnn_cell_phz = tf.contrib.rnn.core_rnn_cell.MultiRNNCell([decoder_rnn_cell_phz] * 2)
            #     decoder_output_phz, decoder_state_phz = tf.nn.dynamic_rnn(decoder_rnn_cell_phz, encoder_output, dtype=tf.float32)
            #
            # with tf.variable_scope("training"):
            #     output = tf.stack([decoder_output_mag, decoder_output_phz], axis=3)
            #     self.loss = tf.reduce_mean(tf.square(output - self.labels))
            #     self.train = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

    def train_model(self, session, inputs, labels):
        session.run(self.train, feed_dict={self.inputs: inputs, self.labels: labels})

    def get_loss(self, session, inputs, labels):
        return session.run(self.loss, feed_dict={self.inputs: inputs, self.labels: labels})

    def embedding_layer(self, vocab_size, embedding_size, input_tensor):
        with tf.device("/cpu:0"):
            emb_weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=0.1))
            embedding_raw = tf.nn.embedding_lookup(emb_weights, input_tensor)
            embedding = tf.expand_dims(embedding_raw, -1)
            return embedding

# m = SpeechSynthesisModel(256, 98, 512, 512)
