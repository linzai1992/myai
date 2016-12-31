from speech_synthesis_model import SpeechSynthesisModel
from gtp_data_batcher import G2PDataBatcher
import tensorflow as tf

print("Loading data...")
batcher = G2PDataBatcher("data/cmudict_proc.txt")

print("Building model...")
model = SpeechSynthesisModel(batcher.sequence_length, len(batcher.word_character_map), len(batcher.phoneme_map))

print("Beginning training")

epochs = 10
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    while True:
        if batcher.epoch_finished():
            input_batches, label_batches = batcher.get_test_batches(50)
            test_acc = 0
            for i in range(len(input_batches)):
                test_acc += model.get_accuracy(session, input_batches[i], label_batches[i])
            test_acc /= len(input_batches)
            print("Epoch ~ %i" % (test_acc))
            batcher.prepare_epoch()

        inputs, labels = batcher.get_training_batch(50)
        model.train_model(session, inputs, labels)
