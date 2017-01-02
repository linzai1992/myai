import tensorflow as tf

class G2PModel():
    def __init__(self, sequence_length, total_chars, total_phons):
        self.sequence_length = sequence_length
        with tf.device("/cpu:0"):
            self.inputs = [tf.placeholder(tf.int32, shape=[None,]) for _ in range(sequence_length)]
            self.labels = [tf.placeholder(tf.int32, shape=[None,]) for _ in range(sequence_length)]
            self.dropout = tf.placeholder(tf.float32)

            cell = tf.nn.rnn_cell.GRUCell(num_units=64)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 2)

            dec_inputs = [tf.ones_like(self.inputs[0], dtype=tf.int32)] + self.labels[:-1] # fill with <GO> token hardcoded for now...
            outputs, states = tf.nn.seq2seq.embedding_rnn_seq2seq(self.inputs, dec_inputs, cell, total_chars, total_phons, 256)

            loss_weights = [tf.ones_like(label, dtype=tf.float32) for label in self.labels]
            loss = tf.nn.seq2seq.sequence_loss(outputs, self.labels, loss_weights, total_phons)
            self.train = tf.train.AdamOptimizer(1e-4).minimize(loss)

    def train_model(self, session, inputs, labels):
        feed_dict = {self.inputs[i]: inputs[i] for i in range(self.sequence_length)}
        feed_dict.update({self.labels[i]: labels[i] for i in range(self.sequence_length)})
        feed_dict.update({self.dropout: 0.5})
        session.run(self.train, feed_dict=feed_dict)

    def get_accuracy(self, session, inputs, labels):
        pass
        # return session.run(self.accuracy, feed_dict={self.inputs: inputs, self.labels: labels, self.sequence_length: [50] * 50})
