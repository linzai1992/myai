from g2p_data_batcher import G2PDataBatcher
from g2p_model import G2PModel
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf

print("Loading data...")
batcher = G2PDataBatcher("data/cmudict_proc.txt")
print("Building model...")
model = G2PModel(batcher.sequence_length, len(batcher.word_character_map), len(batcher.phoneme_map))
saver = tf.train.Saver()
print("Beginning training")
with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    step_index = 0
    epoch_index = 0
    graphemes, phonemes = batcher.get_test_batch()
    loss = model.get_loss(session, graphemes, phonemes)
    print("Initial | loss: %f" % (loss))
    while True:
        if batcher.epoch_finished():
            epoch_index += 1
            graphemes, phonemes = batcher.get_test_batch()
            loss = model.get_loss(session, graphemes, phonemes)
            print("Epoch %i | loss: %f" % (epoch_index, loss))
            saver.save(session, "checkpoints/g2p_model.ckpt")
            batcher.prepare_epoch()

        graphemes, phonemes = batcher.get_training_batch(50)
        model.train_model(session, graphemes, phonemes)

        step_index += 1
        if step_index % 250 == 0:
            graphemes, phonemes = batcher.get_test_batch()
            loss = model.get_loss(session, graphemes, phonemes)
            print("Step %i ~ loss: %f" % (step_index, loss))

        if step_index == 10000:
            break

    while True:
        i = input("Enter word: ").lower().strip()
        p = model.predict(session, batcher.batch_input_string(i))
        for i in range(p.shape[0]):
            word = ""
            for j in range(batcher.sequence_length):
                phon = batcher.phoneme_map_inverse[p[i][j]]
                word += phon + " "
                if phon == "<END>":
                    break
            print(word)
