import os
import numpy as np

class DataBatcher:
    def __init__(self, root_dir_path):
        self.data_file_paths = list()
        self.current_data_file = 0
        self.current_tensor = None
        self.slice_index = 0
        for (dirpath, dirnames, filenames) in os.walk(root_dir_path):
            for filename in filenames:
                if filename.endswith(".npz"):
                    self.data_file_paths.append(os.path.join(dirpath, filename))
            break
        self.test_file_path = self.data_file_paths[-1]
        self.data_file_paths = self.data_file_paths[:-1]
        self.test_tensor = None

    def get_batch(self, size):
        if self.current_tensor is None:
            self.current_tensor = self.__load_tensor_from_file(self.data_file_paths[self.current_data_file])
        if self.slice_index + size >= self.current_tensor.shape[0]:
            size = self.current_tensor.shape[0] - self.slice_index - 1
            # TODO: Get part of next tensor

        batch = self.current_tensor[self.slice_index:self.slice_index + size]
        self.slice_index += size
        epoch_complete = False
        if self.slice_index == self.current_tensor.shape[0] - 1:
            self.slice_index = 0
            self.current_tensor = None
            self.current_data_file += 1
            if self.current_data_file >= len(self.data_file_paths):
                self.current_data_file = 0
                epoch_complete = True
        return batch, epoch_complete

    def get_test_batch(self):
        if self.test_tensor is None:
            self.test_tensor = self.__load_tensor_from_file(self.test_file_path)
        return test_tensor

    def __load_tensor_from_file(self, file_path):
        with np.load(file_path) as data:
            tensors = [t for key, t in data.items()]

        assert len(tensors) > 0, "No tensors loaded from file {}".format(file_path)

        total_tensor = tensors[0]
        if len(tensors) > 1:
            for tensor in tensors[1:]:
                total_tensor = np.concatenate([total_tensor, tensor], axis=0)
        np.random.shuffle(total_tensor)
        total_tensor = np.squeeze(total_tensor, axis=[1, 2])
        return total_tensor
