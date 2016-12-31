import tensorflow as tf


class SpeechSynthesisModel:
    def __init__(self, max_sequence_length, total_chars, total_phons):
        with tf.device("/cpu:0"):
            self.inputs = tf.placeholder(tf.float32, shape=[None, max_sequence_length, total_chars])
            self.labels = tf.placeholder(tf.float32, shape=[None, max_sequence_length, total_phons])
            self.sequence_length = tf.placeholder(tf.int32, shape=[None])

            enc_cell = tf.nn.rnn_cell.LSTMCell(num_units=64)
            enc_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=enc_cell, cell_bw=enc_cell, dtype=tf.float32, sequence_length=self.sequence_length, inputs=self.inputs)
            enc_output_fw, enc_output_bw = enc_outputs
            enc_output_final = enc_output_fw * enc_output_bw

            dec_cell = tf.nn.rnn_cell.LSTMCell(num_units=total_phons)
            dec_outputs, _ = tf.nn.dynamic_rnn(cell=dec_cell, dtype=tf.float32, inputs=enc_output_final)

            self.predictions = tf.argmax(dec_outputs, axis=2)

            loss = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(dec_outputs, self.labels)))
            self.train = tf.train.AdamOptimizer(1e-4).minimize(loss)

            self.accuracy = tf.reduce_mean(tf.reduce_mean(tf.cast(tf.equal(self.predictions, tf.argmax(self.labels, axis=2)), tf.float32)))

    def train_model(self, session, inputs, labels):
        session.run(self.train, feed_dict={self.inputs: inputs, self.labels: labels, self.sequence_length: [50] * 50})

    def get_accuracy(self, session, inputs, labels):
        return session.run(self.accuracy, feed_dict={self.inputs: inputs, self.labels: labels, self.sequence_length: [50] * 50})

    def synthesize(self):
        # beep boop
        pass
