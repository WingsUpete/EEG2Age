import os
import sys
import time

import torch
from torch.utils.data import Dataset, DataLoader

import Config

# https://stackoverflow.com/questions/50307707/convert-pandas-dataframe-to-pytorch-tensor


# def checkFileExistence(path):
#     if not os.path.isfile(path):
#         sys.stderr.write('[DataSet] %s does not exist.\n' % path)
#         exit(-50)
#     return


def checkKFold(folds, k):
    if k != -1 and (k < 0 or k >= folds):
        sys.stderr.write('[DataSet] k should be -1 or in range [0, folds) and k is integer; however, k = %d and folds = %d breaks the condition\n' % (
            k, folds
        ))
        exit(-100)


class EEGAgeDataSetItem(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


# TODO
class EEGAgeDataSet:
    def __init__(self, data_dir, folds=5, valid_k=-1):
        self.data_dir = data_dir

        self.train_set = None
        self.valid_set = None
        self.test_set = None


def evalSamplingSpeed(ds, batch_size, shuffle, tag, num_workers=4):
    """
    Test the sampling functionality & efficiency
    """
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    time0 = time.time()

    for i, batch in enumerate(dataloader):
        # features, labels = batch['features'], batch['target']
        sys.stderr.write("\r{} Set - Batch No. {}/{} with time used(s): {}".format(tag, i + 1, len(dataloader), time.time() - time0))
        sys.stderr.flush()

    sys.stderr.write("\n")


if __name__ == '__main__':
    # TODO: Test sampling dataset
    pass
