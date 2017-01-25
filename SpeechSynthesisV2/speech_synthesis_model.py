import tensorflow as tf

class SpeechSynthesisModel:
    def __init__(self, sequence_length, vocab_size, embedding_size, output_shape):
        assert len(output_shape) == 2, "output_shape must have length 2"

        with tf.variable_scope("speech_synthesizer"):
            self.inputs = tf.placeholder(tf.int32, shape=[None, sequence_length])
            self.labels = tf.placeholder(tf.float32, shape=[None] + output_shape + [2])

            embeddings = self.embedding_layer(vocab_size, embedding_size, self.inputs)

            # Encoding layers
            W_conv1 = tf.Variable(tf.truncated_normal([3, 3, 1, 32], stddev=0.1))
            b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
            c_conv1 = tf.nn.conv2d(embeddings, W_conv1, strides=[1, 1, 1, 1], padding="SAME")
            a_conv1 = tf.nn.elu(c_conv1 + b_conv1)

            # Sequencing layer
            filter_height = 3
            a_conv1_padded = tf.pad(a_conv1, [[0,0], [(filter_height-1)//2,(filter_height-1)//2], [0,0], [0,0]], "CONSTANT")
            W_conv2 = tf.Variable(tf.truncated_normal([filter_height, embedding_size, 32, 32], stddev=0.1))
            b_conv2 = tf.Variable(tf.constant(0.1, shape=[32]))
            c_conv2 = tf.nn.conv2d(a_conv1_padded, W_conv2, strides=[1, 1, 1, 1], padding="VALID")
            a_conv2 = tf.nn.elu(c_conv2 + b_conv2)

            # Decoding layers
            W_conv3 = tf.Variable(tf.truncated_normal([3, 1, 128, 32], stddev=0.1))
            b_conv3 = tf.Variable(tf.constant(0.1, shape=[128]))
            # c_conv3 = tf.nn.conv2d_transpose(a_conv2, W_conv3, [-1] + [i//2 for i in output_shape] + [128], strides=[1, 1, 256, 1], padding="SAME")
            c_conv3 = tf.nn.conv2d_transpose(a_conv2, W_conv3, [30, output_shape[0]//2, output_shape[1]//2, 128], strides=[1, 1, 256, 1], padding="SAME")
            a_conv3 = tf.nn.elu(c_conv3 + b_conv3)

            W_conv4 = tf.Variable(tf.truncated_normal([3, 3, 2, 128], stddev=0.1))
            b_conv4 = tf.Variable(tf.constant(0.1, shape=[2]))
            c_conv4 = tf.nn.conv2d_transpose(a_conv3, W_conv4, [30] + output_shape + [2], strides=[1, 2, 2, 1], padding="SAME")
            a_conv4 = tf.nn.elu(c_conv4 + b_conv4)

            self.output = a_conv4
            # TODO: Add batch norm layers

            # Training layers
            self.loss = tf.reduce_mean(tf.square(self.output - self.labels))
            self.train = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

    def generate_spectral_features(self, session, inputs):
        return session.run(self.output, feed_dict={self.inputs: inputs})

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

# m = SpeechSynthesisModel(sequence_length=10, vocab_size=80, embedding_size=50, output_shape=[512, 512])
