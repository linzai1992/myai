from nlp_model import NLPModel
from data_batcher import DataBatcher
import tensorflow as tf

batcher = DataBatcher("data/command_data_noisy.json")
model = NLPModel((batcher.max_sentence_len, batcher.total_words), batcher.total_classes, [3,4,5])

saver = tf.train.Saver()

with tf.Session() as session:
	session.run(tf.global_variables_initializer())
	epoch_index = 0
	for i in range(10000):
		if batcher.epoch_finished():
			sentences, labels = batcher.generate_full_batch()
			accuracy = model.get_accuracy(session, sentences, labels)
			print("Epoch %i ~ %f" % (epoch_index, accuracy))
			epoch_index += 1
			batcher.prepare_epoch()
			saver.save(session, "checkpoints/nlp_model.ckpt")
		else:
			sentences, labels = batcher.generate_batch(50)
			model.train(session, sentences, labels)
	saver.save(session, "checkpoints/nlp_model.ckpt")