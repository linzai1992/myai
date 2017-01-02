import tensorflow as tf

class G2PModel():
    def __init__(self, max_sequence_length, total_chars, total_phons):
        with tf.device("/cpu:0"):
            self.inputs = tf.placeholder(tf.float32, shape=[None, max_sequence_length, total_chars])
            self.labels = tf.placeholder(tf.float32, shape=[None, max_sequence_length, total_phons])
            self.dropout = tf.placeholder(tf.float32)

            cell = tf.nn.rnn_cell.GRUCell(num_units=64)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_probability=self.dropout)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 2)

            dec_inputs = tf.ones([total_phons]) +
            outputs, states = tf.nn.seq2seq.embedding_rnn_seq2seq(self.inputs, dec_inputs, cell, total_chars, total_phons, 256)
            

    def train_model(self, session, inputs, labels):
        pass
        # session.run(self.train, feed_dict={self.inputs: inputs, self.labels: labels, self.sequence_length: [50] * 50})

    def get_accuracy(self, session, inputs, labels):
        pass
        # return session.run(self.accuracy, feed_dict={self.inputs: inputs, self.labels: labels, self.sequence_length: [50] * 50})
