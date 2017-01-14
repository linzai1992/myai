import tensorflow as tf

class VAEModel:
    def __init__(self, window_size, layers):
        self.inputs = tf.placeholder(tf.float32, shape=[None, window_size])

        # Encoding block
        last_size = window_size
        last_layer = self.inputs
        for layer_size in layers[:-1]: # [60 40]
            weights = tf.Variable(tf.truncated_normal([last_size, layer_size], stddev=0.1))
            biases = tf.Variable(tf.constant(0.1, shape=[layer_size]))
            last_layer = tf.nn.elu(tf.nn.xw_plus_b(last_layer, weights, biases))
            last_size = layer_size

        layers_reversed = layers[:-1]
        layers_reversed.reverse()
        layers_reversed.append(window_size)

        # Latent block
        W_mu_block = tf.Variable(tf.truncated_normal([last_size, layers_reversed[0]], stddev=0.1))
        b_mu_block = tf.Variable(tf.constant(0.1, shape=[layers_reversed[0]]))
        mu_block = tf.nn.xw_plus_b(last_layer, W_mu_block, b_mu_block)

        W_sigma_block = tf.Variable(tf.truncated_normal([last_size, layers_reversed[0]], stddev=0.1))
        b_sigma_block = tf.Variable(tf.constant(0.1, shape=[layers_reversed[0]]))
        sigma_block = tf.nn.xw_plus_b(last_layer, W_sigma_block, b_sigma_block)

        samples = tf.random_normal(tf.shape(sigma_block), 0.0, 1.0, tf.float32)
        z = mu_block + (sigma_block * samples)
        # TODO: Perhaps add non-linearity here
        last_layer = z
        last_size = layers_reversed[0]

        # Decoding block
        for layer_size in layers_reversed:
            weights = tf.Variable(tf.truncated_normal([last_size, layer_size], stddev=0.1))
            biases = tf.Variable(tf.constant(0.1, shape=[layer_size]))
            last_layer = tf.nn.elu(tf.nn.xw_plus_b(last_layer, weights, biases))
            last_size = layer_size

        generation_loss = tf.reduce_mean(tf.square(self.inputs - last_layer), axis=1)
        latent_loss = 0.5 * tf.reduce_sum(tf.square(mu_block) + tf.square(sigma_block) - tf.log(tf.square(sigma_block)) - 1.0, axis=1)
        self.loss = tf.reduce_sum(generation_loss + latent_loss)
        self.train = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

    def train_model(self, session, inputs):
        session.run(self.train, feed_dict={self.inputs: inputs})

    def get_loss(self, session, inputs):
        return session.run(self.loss, feed_dict={self.inputs: inputs})
