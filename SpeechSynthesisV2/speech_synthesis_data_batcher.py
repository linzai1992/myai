import numpy as np

class SpeechSynthesisDataBatcher:
    def __init__(self, train_data_dir_path, test_data_dir_path):
        self.train_data_dir_path = train_data_dir_path
        self.test_data_dir_path = test_data_dir_path
        self.test_file_index = 0
        self.current_test_batch = None

    def get_batch(self, size):
        if self.current_test_batch is None:
            # load next one
            pass

    def load_batch_from_file(self, file_path):
        with open(file_path, "rb") as batch_file:
            batch_dict = np.load(batch_file)
            print(batch_dict.keys())
            print(batch_dict["arr_1"].shape)

s = SpeechSynthesisDataBatcher("data/generated/training_samples", None)
s.load_batch_from_file("data/generated/training_samples/sample_0.npz")
