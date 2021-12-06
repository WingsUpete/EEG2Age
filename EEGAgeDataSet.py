import torch
from torch.utils.data import Dataset, DataLoader

import Config


# TODO
class EEGAgeDataSet(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.train_set = None
        self.valid_set = None
        self.test_set = None

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return None


if __name__ == '__main__':
    # TODO: Test sampling dataset
    pass
