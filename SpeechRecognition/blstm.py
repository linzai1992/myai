import tensorflow as tf

def BLSTMCell(lstm_hidden, batch_size):
	cell_fw = tf.nn.rnn_cell.LSTMCell(lstm_hidden, use_peepholes=True, state_is_tuple=True)
	cell_bw = tf.nn.rnn_cell.LSTMCell(lstm_hidden, use_peepholes=True, state_is_tuple=True)
	initial_state_fw = cell_fw.zero_state(batch_size, tf.float32)
	initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)
	return cell_fw, cell_bw, initial_state_fw, initial_state_bw

# def BLSTM(lstm_hidden, X, chunck_size, name="BLSTM", seq_len=None):
# 	# initializer = tf.random_uniform_initializer(-1, 1)
# 	cell_fw = tf.nn.rnn_cell.LSTMCell(lstm_hidden, initializer=initializer, use_peepholes=True, state_is_tuple=True)
# 	cell_bw = tf.n..rnn_cell.LSTMCell(lstm_hidden, initializer=initializer, use_peepholes=True, state_is_tuple=True)

# 	initial_state_fw = cell_fw.zero_state(batch_size, tf.float32)
# 	initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)

# 	if seq_len is not None:
# 		output, _, _ = tf.nn.bidirectional_rnn(cell_fw, cell_bw, X, initial_state_fw=initial_state_fw, initial_state_bw=initial_state_bw, sequence_length=seq_len, scope=name)
# 	else:
# 		output, _, _ = tf.nn.bidirectional_rnn(cell_fw, cell_bw, X, initial_state_fw=initial_state_fw, initial_state_bw=initial_state_bw, scope=name)

# 	return output