import tensorflow as tf

class SpeechSynthesisModel:

    def __init__(self, input_seq_length, char_embedding_size):
        with tf.variable_scope("SpeechSynthesisModelTacotron"):
            self.inputs = tf.placeholder(tf.float32, shape=[None, input_seq_length, char_embedding_size, 1])
            kek = self.CBHG(self.inputs)

            # ins = tf.placeholder(tf.float32, shape=[None, 128])
            # hi = self.highway(ins)

    def CBHG(self, input_tensor, k=16, in_channels=1, out_channels=128):
        with tf.variable_scope("CBHG"):
            tensor_width = int(input_tensor.get_shape()[-2])
            conv_bank_outputs = list()
            top_pad_amount = 0
            bottom_pad_amount = 0
            for i in range(1, k+1):
                padded_input = tf.pad(input_tensor, paddings=[[0,0], [top_pad_amount,bottom_pad_amount], [0,0], [0,0]], mode="CONSTANT", name="padded_conv1d_bank_input_{}".format(i))
                print(padded_input.get_shape())
                if top_pad_amount == bottom_pad_amount: top_pad_amount += 1
                elif top_pad_amount > bottom_pad_amount: bottom_pad_amount += 1

                w = tf.Variable(tf.truncated_normal([i, tensor_width, in_channels, out_channels], stddev=0.1))
                b = tf.Variable(tf.constant(0.1, shape=[out_channels]))
                conv = tf.nn.conv2d(padded_input, filter=w, strides=[1,1,1,1], padding="VALID")
                act = tf.nn.elu(conv + b, name="1d_conv_bank_{}".format(i))
                conv_bank_outputs.append(act)

            print("Banks out =========================")
            for act in conv_bank_outputs:
                print(act.get_shape())

            print("Concat banks =========================")
            concatenated_banks = tf.concat(conv_bank_outputs, axis=-2, name="concatenated_banks")
            print(concatenated_banks.get_shape())

            print("Pooled banks =========================")
            padded_concat_banks = tf.pad(concatenated_banks, paddings=[[0,0], [1,0], [0,0], [0,0]], mode="CONSTANT", name="padded_concat_banks")
            max_pooled_banks = tf.nn.max_pool(padded_concat_banks, ksize=[1,2,k,1], strides=[1,1,1,1], padding="VALID", name="pooled_banks")
            print(max_pooled_banks.get_shape())

            print("Unfurled banks =========================")
            unfurled_banks = tf.transpose(max_pooled_banks, perm=[0,1,3,2])
            padded_unfurled_banks = tf.pad(unfurled_banks, paddings=[[0,0], [1,1], [0,0], [0,0]], mode="CONSTANT", name="padded_unfurled_banks")
            print(unfurled_banks.get_shape())

            print("Projections =========================")
            projections_act = list()
            for _ in range(3):
                w = tf.Variable(tf.truncated_normal([3, out_channels, 1, out_channels], stddev=0.1))
                b = tf.Variable(tf.constant(0.1, shape=[out_channels]))
                conv = tf.nn.conv2d(padded_unfurled_banks, filter=w, strides=[1,1,1,1], padding="VALID")
                act = tf.nn.elu(conv + b)
                projections_act.append(act)

            concatenated_projections_act = tf.concat(projections_act, axis=-2, name="concatenated_projections_act")
            padded_projections_act = tf.pad(concatenated_projections_act, paddings=[[0,0], [1,1], [0,0], [0,0]], mode="CONSTANT")
            embedding_size = int(input_tensor.get_shape()[-2])
            for _ in range(3):
                w = tf.Variable(tf.truncated_normal([3, 3, out_channels, embedding_size], stddev=0.1))
                b = tf.Variable(tf.constant(0.1, shape=[out_channels]))
                conv = tf.nn.conv2d(padded_projections_act, filter=w, strides=[1,1,1,1], padding="VALID")
                conv_residual = tf.transpose(conv, perm=[0,1,3,2]) + input_tensor
                print(conv_residual.get_shape())

            # pooled_banks = tf.nn.max_pool(concatenated_banks, ksize=[1,2,tensor_width,1], strides=[1,1,1,1], padding="SAME", name="pooled_banks")

            # projections = list()
            # for i in range(3):
            #     w = tf.Variable(tf.truncated_normal([3, tensor_width, out_channels * k, out_channels], stddev=0.1))
            #     b = tf.Variable(tf.constant(0.1, shape=[out_channels]))
            #     conv = tf.nn.conv2d(pooled_banks, filter=w, strides=[1,1,1,1], padding="SAME")
            #     act = tf.nn.elu(conv + b, name="1d_conv_projection_{}".format(i+1))
            #     projections.append(act)
            #
            # print(input_tensor.get_shape())
            # print("Final =========================")
            # for p in projections:
            #     print(p.get_shape())

    def highway(self, input_tensor, layers=4, hidden_size=128):
        with tf.variable_scope("Highway"):
            last_input = input_tensor
            for _ in range(layers):
                size = int(last_input.get_shape()[-1])

                w_h = tf.Variable(tf.truncated_normal([size, hidden_size], stddev=0.1))
                b_h = tf.Variable(tf.constant(0.1, shape=[hidden_size]))
                h_x = tf.nn.elu(tf.nn.xw_plus_b(last_input, w_h, b_h))

                w_t = tf.Variable(tf.truncated_normal([size, hidden_size], stddev=0.1))
                b_t = tf.Variable(tf.constant(0.1, shape=[hidden_size]))
                t_x = tf.nn.sigmoid(tf.nn.xw_plus_b(last_input, w_t, b_t))
                c_x = 1.0 - t_x

                y = (h_x * t_x) + (last_input * c_x)
                last_input = y
            return last_input

m = SpeechSynthesisModel(100, 50)
