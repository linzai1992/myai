import tensorflow as tf

class SpeechSynthesisModel:
    def __init__(self, max_sequence_length, total_chars, total_phons):
        with tf.device("/cpu:0"):
            pass

    def train_model(self, session, inputs, labels):
        pass

    def get_accuracy(self, session, inputs, labels):
        pass

    def synthesize(self):
        # beep boop
        pass
