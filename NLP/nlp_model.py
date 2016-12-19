import tensorflow as tf

class NLPModel:
	def __init__(self, input_shape, output_classes, filters):
		self.input = tf.placeholder(tf.float32, shape=[None, input_shape[0], input_shape[1], 1])
		self.labels = tf.placeholder(tf.float32, shape=[None, output_classes])
		self.dropout = tf.placeholder(tf.float32)

		filter_outputs = []
		for index, filter_length in enumerate(filters):
			W = tf.Variable(tf.truncated_normal([filter_length, input_shape[1], 1, 32], stddev=0.1))
			b = tf.Variable(tf.constant(0.1, shape=[32]))
			conv_l1 = tf.nn.conv2d(self.input, W, strides=[1,1,1,1], padding="VALID")
			conv_pool = tf.nn.elu(tf.nn.bias_add(conv_l1, b))
			conv_pool = tf.nn.max_pool(conv_pool, ksize=[1, input_shape[0] - filter_length + 1, 1, 1], strides=[1,1,1,1], padding="VALID")
			# conv_pool = tf.reshape(conv_pool, [-1, 32])
			# drop = tf.nn.dropout(conv_pool, self.dropout)
			filter_outputs.append(conv_pool)

		total_filters = 32 * len(filters)
		final_pool = tf.concat(3, filter_outputs)
		final_pool_flat = tf.reshape(final_pool, [-1, total_filters])
		drop = tf.nn.dropout(final_pool_flat, self.dropout)

		W_fc = tf.Variable(tf.truncated_normal([total_filters, output_classes], stddev=0.1))
		b_fc = tf.Variable(tf.constant(0.1, shape=[output_classes]))
		raw_scores = tf.nn.xw_plus_b(drop, W_fc, b_fc)
		self.probabilities = tf.nn.softmax(raw_scores)
		self.predictions = tf.argmax(self.probabilities, 1)

		self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predictions, tf.argmax(self.labels, 1)), tf.float32))

		self.loss = tf.nn.softmax_cross_entropy_with_logits(raw_scores, self.labels)
		self.optimize = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

	def predict(self, session, inputs):
		return session.run(self.predictions, feed_dict={self.input: inputs, self.dropout: 1.0})[0]

	def train(self, session, inputs, labels):
		session.run(self.optimize, feed_dict={self.input: inputs, self.labels: labels, self.dropout: 0.5})

	def get_accuracy(self, session, inputs, labels):
		return session.run(self.accuracy, feed_dict={self.input: inputs, self.labels: labels, self.dropout: 1.0})