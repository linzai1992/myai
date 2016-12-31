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
            print("finished an epoch!")
            batcher.prepare_epoch()

        print("Trained!")
        inputs, labels = batcher.get_training_batch(50)
        model.train_model(session, inputs, labels)
