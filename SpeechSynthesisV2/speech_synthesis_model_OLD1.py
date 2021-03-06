# import tensorflow as tf
#
# class SpeechSynthesisModel:
#     def __init__(self, sequence_length, vocab_size, embedding_size, output_shape):
#         assert len(output_shape) == 2, "output_shape must have length 2"
#
#         with tf.variable_scope("speech_synthesizer"), tf.device("/cpu:0"):
#             self.inputs = tf.placeholder(tf.int32, shape=[None, sequence_length])
#             self.batch_size = tf.placeholder(tf.int32)
#             self.labels = tf.placeholder(tf.float32, shape=[None] + output_shape + [2])
#
#             embeddings = self.embedding_layer(vocab_size, embedding_size, self.inputs)
#
#             # Encoding layers
#             W_conv1 = tf.Variable(tf.truncated_normal([3, 3, 1, 64], stddev=0.1))
#             b_conv1 = tf.Variable(tf.constant(0.1, shape=[64]))
#             c_conv1 = tf.nn.conv2d(embeddings, W_conv1, strides=[1, 1, 1, 1], padding="SAME")
#             a_conv1 = tf.nn.elu(c_conv1 + b_conv1)
#
#             # Sequencing layer
#             # filter_height = 3
#             # a_conv1_padded = tf.pad(a_conv1, [[0,0], [(filter_height-1)//2,(filter_height-1)//2], [0,0], [0,0]], "CONSTANT")
#             # W_conv2 = tf.Variable(tf.truncated_normal([filter_height, embedding_size, 64, 64], stddev=0.1))
#             # b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
#             # c_conv2 = tf.nn.conv2d(a_conv1_padded, W_conv2, strides=[1, 1, 1, 1], padding="VALID")
#             # a_conv2 = tf.nn.elu(c_conv2 + b_conv2)
#
#             # Decoding layers
#             W_conv3 = tf.Variable(tf.truncated_normal([3, 1, 128, 64], stddev=0.1))
#             b_conv3 = tf.Variable(tf.constant(0.1, shape=[128]))
#             c_conv3 = tf.nn.conv2d_transpose(a_conv1, W_conv3, [self.batch_size, output_shape[0]//2, output_shape[1]//2, 128], strides=[1, 1, 4, 1], padding="SAME")
#             a_conv3 = tf.nn.elu(c_conv3 + b_conv3)
#
#             W_conv4 = tf.Variable(tf.truncated_normal([3, 3, 2, 128], stddev=0.1))
#             b_conv4 = tf.Variable(tf.constant(0.1, shape=[2]))
#             c_conv4 = tf.nn.conv2d_transpose(a_conv3, W_conv4, [self.batch_size, output_shape[0], output_shape[1], 2], strides=[1, 2, 2, 1], padding="SAME")
#             a_conv4 = tf.nn.elu(c_conv4 + b_conv4)
#
#             self.output = a_conv4
#             # TODO: Add batch norm layers
#
#             # Training layers
#             self.loss = tf.reduce_sum(tf.reduce_mean(tf.square(self.output - self.labels), axis=[1,2,3]))
#             self.train = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
#
#     def generate_spectral_features(self, session, inputs, batch_size):
#         return session.run(self.output, feed_dict={self.inputs: inputs, self.batch_size: batch_size})
#
#     def train_model(self, session, inputs, labels, batch_size):
#         session.run(self.train, feed_dict={self.inputs: inputs, self.labels: labels, self.batch_size: batch_size})
#
#     def get_loss(self, session, inputs, labels, batch_size):
#         return session.run(self.loss, feed_dict={self.inputs: inputs, self.labels: labels, self.batch_size: batch_size})
#
    # def embedding_layer(self, vocab_size, embedding_size, input_tensor):
    #     with tf.device("/cpu:0"):
    #         emb_weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=0.1))
    #         embedding_raw = tf.nn.embedding_lookup(emb_weights, input_tensor)
    #         embedding = tf.expand_dims(embedding_raw, -1)
    #         return embedding

# m = SpeechSynthesisModel(sequence_length=10, vocab_size=80, embedding_size=50, output_shape=[512, 512])
