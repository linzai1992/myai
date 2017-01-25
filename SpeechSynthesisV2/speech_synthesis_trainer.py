import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from speech_synthesis_data_batcher import SpeechSynthesisDataBatcher
from speech_synthesis_model import SpeechSynthesisModel
import tensorflow as tf

train_data_path = os.path.join("data", "generated", "development_samples")
test_data_path = os.path.join("data", "generated", "test_samples")
phone_map_path = os.path.join("data", "generated", "phone_map.txt")

batcher = SpeechSynthesisDataBatcher(train_data_path, test_data_path)
print("Building model...")
model = SpeechSynthesisModel(sequence_length=256, vocab_size=batcher.get_vocab_size(phone_map_path), embedding_size=64, output_shape=[512, 512])
saver = tf.train.Saver()

epochs = 10
batch_size = 30

print("Beginning training...")
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    current_epoch = 0
    current_step = 0
    while current_epoch < epochs:
        phones_batch, spect_batch, epoch_complete = batcher.get_batch(batch_size)
        model.train_model(session, inputs=phones_batch, labels=spect_batch, batch_size=phones_batch.shape[0])

        if epoch_complete:
            avg_acc = 0.0
            count = 0
            for phones_test_batch, spect_test_batch in batcher.get_test_batches():
                avg_acc += model.get_loss(session, inputs=phones_test_batch, labels=spect_test_batch, batch_size=phones_test_batch.shape[0])
                count += 1
            avg_acc /= count
            saver.save(session, os.path.join("checkpoints", "speech_synthesis_model"))
            print("Epoch {} | loss: {}".format(current_epoch+1, avg_acc))
