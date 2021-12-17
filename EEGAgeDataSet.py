import os
import sys
import time

import torch
from torch.utils.data import Dataset, DataLoader

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import dgl
sys.stderr.close()
sys.stderr = stderr

import Config


def checkPathExistence(path):
    if not os.path.exists(path):
        sys.stderr.write('[DataSet] %s does not exist.\n' % path)
        exit(-50)
    return


def checkKFold(folds, k):
    if k != -1 and (k < 0 or k >= folds):
        sys.stderr.write('[DataSet] k should be -1 or in range [0, folds) and k is integer; however, k = %d and folds = %d breaks the condition\n' % (
            k, folds
        ))
        exit(-100)


class EEGAgeDataSetItem(Dataset):
    def __init__(self, data_dir, data_ids: list):
        self.data_dir = data_dir
        self.data_ids = data_ids

        self.graph_path = os.path.join(self.data_dir, 'graph.dgl')

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        # 1. Load features and target from sample
        curFile = os.path.join(self.data_dir, '%d.pt' % self.data_ids[idx])
        checkPathExistence(curFile)
        curPack = torch.load(curFile)
        V, target = curPack['V'], curPack['target']

        # 2. Load graph
        (graph,), _ = dgl.load_graphs(self.graph_path)
        graph.ndata['v'] = V

        return {
            'inputs': {
                'features': V,
                'graph': graph,
            },
            'target': target
        }


class EEGAgeDataSet:
    def __init__(self, data_dir, n_samples, folds=5, valid_k=-1):
        self.data_dir = data_dir
        checkPathExistence(self.data_dir)
        self.n_samples = n_samples

        self.folds = folds
        self.valid_k = valid_k
        checkKFold(self.folds, self.valid_k)

        n_test = int(self.n_samples * 0.2)
        n_train_valid = self.n_samples - n_test

        self.train_set_ids, self.valid_set_ids = self.train_valid_split(n_train_valid)
        self.test_set_ids = [x + 1 for x in range(n_train_valid, self.n_samples)]

        # DataSet
        self.train_set = EEGAgeDataSetItem(data_dir=self.data_dir, data_ids=self.train_set_ids)
        self.valid_set = EEGAgeDataSetItem(data_dir=self.data_dir, data_ids=self.valid_set_ids)
        self.test_set = EEGAgeDataSetItem(data_dir=self.data_dir, data_ids=self.test_set_ids)

    def train_valid_split(self, n_train_valid):
        n_samples_per_fold = int(n_train_valid / self.folds)
        steps = [n_samples_per_fold * i for i in range(self.folds)]
        train, valid = [], []
        for i in range(len(steps) - 1):
            curFold = [x + 1 for x in range(steps[i], steps[i + 1])]
            if self.valid_k == i:
                valid += curFold
            else:
                train += curFold
        if self.valid_k == self.folds - 1 or self.valid_k == -1:
            valid += [x + 1 for x in range(steps[-1], n_train_valid)]
        else:
            train += [x + 1 for x in range(steps[-1], n_train_valid)]

        return train, valid


def evalSamplingSpeed(ds, batch_size, shuffle, tag, num_workers=4):
    """
    Test the sampling functionality & efficiency
    """
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    time0 = time.time()

    for i, batch in enumerate(dataloader):
        features, graph, label = batch['inputs']['features'], batch['inputs']['graph'], batch['target']
        sys.stderr.write("\r{} Set - Batch No. {}/{} with time used(s): {}".format(tag, i + 1, len(dataloader), time.time() - time0))
        sys.stderr.flush()

    sys.stderr.write("\n")


if __name__ == '__main__':
    dataset = EEGAgeDataSet(data_dir=Config.DATA_DIR_DEFAULT, n_samples=Config.NUM_SAMPLES, folds=5, valid_k=-1)
    print(dataset)
    # evalSamplingSpeed(dataset.train_set, batch_size=5, shuffle=True, tag='Training', num_workers=4)
    # evalSamplingSpeed(dataset.valid_set, batch_size=3, shuffle=False, tag='Validation', num_workers=4)
    # evalSamplingSpeed(dataset.test_set, batch_size=3, shuffle=False, tag='Test', num_workers=4)
