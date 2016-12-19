from data_batcher import DataBatcher
from nlp_model import NLPModel
import tensorflow as tf

batcher = DataBatcher("data/command_data_noisy.json")
model = NLPModel((batcher.max_sentence_len, batcher.total_words), batcher.total_classes, [3,4,5])

saver = tf.train.Saver()

with tf.Session() as session:
	session.run(tf.global_variables_initializer())
	ckpt = tf.train.get_checkpoint_state("checkpoints/")
	if ckpt and ckpt.model_checkpoint_path:
		saver.restore(session, ckpt.model_checkpoint_path)
	else:
		print("Failed loading NLP model!")

	print(batcher.key_map)
	while True:
		command = input("› ")
		command_processed = batcher.preprocess_string(command)
		tensor = batcher.sentence_to_tensor(command_processed)
		prediction = model.predict(session, tensor)
		print("\tPredicted %s" % batcher.index_key_map[prediction])