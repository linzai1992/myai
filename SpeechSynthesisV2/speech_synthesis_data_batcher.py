import os
import numpy as np

class SpeechSynthesisDataBatcher:
    def __init__(self, train_data_dir_path, test_data_dir_path):
        self.train_batch_paths = self.get_batch_file_paths(train_data_dir_path)
        self.test_batch_paths = self.get_batch_file_paths(test_data_dir_path)
        self.train_file_index = 0
        self.test_file_index = 0
        self.train_batch_index = 0
        self.current_train_batch = None
        self.current_test_batch = None

    def get_batch(self, size):
        if self.current_train_batch is None:
            self.current_train_batch = self.load_batch_from_file(self.train_batch_paths[self.train_file_index])
            self.train_file_index += 1

        if self.train_batch_index + size > self.current_train_batch[0].shape[0]:
            size = self.current_train_batch[0].shape[0] - self.train_file_index

        phones_tensor = self.current_train_batch[0][self.train_batch_index:self.train_batch_index + size]
        spect_tensor = self.current_train_batch[1][self.train_batch_index:self.train_batch_index + size]

        epoch_complete = False
        self.train_batch_index += size
        if self.train_batch_index >= self.current_train_batch[0].shape[0]:
            self.train_file_index += 1
            self.current_train_batch = None
            if self.train_file_index >= len(self.train_batch_paths):
                self.train_file_index = 0
                epoch_complete = True

        return phones_tensor, spect_tensor, epoch_complete

    def get_test_batches(self):
        for path in self.test_batch_paths:
            phones_tensor, spect_tensor = self.load_batch_from_file(path)
            yield phones_tensor, spect_tensor

    def load_batch_from_file(self, file_path):
        with open(file_path, "rb") as batch_file:
            batch_dict = np.load(batch_file)
            phone_data = batch_dict["arr_0"]
            spect_data = batch_dict["arr_1"]
            return (phone_data, spect_data)

    def get_batch_file_paths(self, root_dir_path):
        for dirpath, dirnames, filenames in os.walk(root_dir_path):
            return [os.path.join(dirpath, name) for name in filenames if name.endswith(".npz")]
            break

    def get_vocab_size(self, path):
        count = 0
        with open(path, "r") as map_file:
            for line in map_file:
                count += 1
        return count

# s = SpeechSynthesisDataBatcher("data/generated/development_samples", "data/generated/development_samples")
# test_gen = s.get_test_batches()
# for p, sp in test_gen:
#     print(p.shape, sp.shape)
