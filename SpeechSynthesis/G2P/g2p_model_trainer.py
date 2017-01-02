from g2p_data_batcher import G2PDataBatcher
from g2p_model import G2PModel
import tensorflow as tf

print("Loading data...")
batcher = G2PDataBatcher("data/cmudict_proc.txt")
print("Building model...")
model = G2PModel(batcher.sequence_length, len(batcher.word_character_map), len(batcher.phoneme_map))

print("Beginning training")
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    while True:
        print("step")
        graphemes, phonemes = batcher.get_training_batch(5)
        model.train_model(session, graphemes, phonemes)
