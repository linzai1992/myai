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

    step_index = 0
    while True:
        graphemes, phonemes = batcher.get_training_batch(5)
        model.train_model(session, graphemes, phonemes)

        step_index += 1
        if step_index % 40 == 0:
            graphemes, phonemes = batcher.get_test_batch()
            loss = model.get_loss(session, graphemes, phonemes)
            print("Step %i ~ loss: %f" % (step_index, loss))

        if step_index == 1000:
            break

    while True:
        i = input("Enter word: ").lower().strip()
        p = model.predict(session, batcher.batch_input_string(i))
        for i in range(p.shape[0]):
            word = ""
            for j in range(batcher.sequence_length):
                word += batcher.phoneme_map_inverse[p[i][j]] + " "
            print(word)
