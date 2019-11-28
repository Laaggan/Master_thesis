import keras
import numpy as np
from my_lib import *

class Generator(keras.utils.Sequence):

    def __init__(self, indices, path_to_folder, batch_size, modality, cropping=True):
        self.path = path_to_folder
        self.cropping = cropping
        self.indices = indices
        self.batch_size = batch_size
        self.rand_batch = np.random.choice(indices, batch_size)
        self.modality = modality

    def __len__(self):
        'Denotes the number of batches per epoch'
        return (np.ceil(len(self.indices) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, index):
        x, y = load_patients_numpy(
            path_to_folder=self.path,
            indices=self.rand_batch,
            cropping=self.cropping
        )
        x = x[:, :, :, self.modality]
        H = x.shape[1]
        W = x.shape[2]
        x = x.reshape(-1, H, W, 1)
        return x, y
