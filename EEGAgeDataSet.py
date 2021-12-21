import os
import sys
import time
import random

import torch

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import dgl
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
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


class EEGAgeDataSetItem(DGLDataset):
    def __init__(self, data_dir, data_ids: list, cust_graph: bool):
        self.data_dir = data_dir
        self.data_ids = data_ids
        self.cust_graph = cust_graph

        self.graph_path = os.path.join(self.data_dir, 'graph_customize.dgl' if self.cust_graph else 'graph.dgl')

    def process(self):
        pass

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
    def __init__(self, data_dir, n_samples, sample_split=Config.SAMPLE_SPLIT, cust_graph=False,
                 folds=Config.FOLDS_DEFAULT, valid_k=Config.VALID_K_DEFAULT):
        self.data_dir = data_dir
        checkPathExistence(self.data_dir)

        self.cust_graph = cust_graph

        self.n_samples = n_samples  # total: 1 -> n
        self.sample_split = sample_split
        self.bad_samples = self.getBadSamples()
        self.n_bad_samples = len(self.bad_samples)
        self.n_valid_samples = self.n_samples - self.n_bad_samples

        # Randomly shuffle the samples
        random.seed(Config.SHUFFLE_RANDOM_SEED)
        self.real_sample_map = self.getValidSamples()
        random.shuffle(self.real_sample_map)
        random.seed(None)

        self.folds = folds
        self.valid_k = valid_k
        checkKFold(self.folds, self.valid_k)

        n_test = int(self.n_valid_samples * 0.2)
        n_train_valid = self.n_valid_samples - n_test

        # Fake IDs which only specify the order
        self.train_set_ids, self.valid_set_ids = self.train_valid_split(n_train_valid)
        self.test_set_ids = [x for x in range(n_train_valid, self.n_valid_samples)]
        # Find the real id for the sample
        self.train_set_ids = [self.real_sample_map[train_set_id] for train_set_id in self.train_set_ids]
        self.valid_set_ids = [self.real_sample_map[valid_set_id] for valid_set_id in self.valid_set_ids]
        self.test_set_ids = [self.real_sample_map[test_set_id] for test_set_id in self.test_set_ids]

        # DataSet
        self.train_set = EEGAgeDataSetItem(data_dir=self.data_dir, data_ids=self.train_set_ids, cust_graph=self.cust_graph)
        self.valid_set = EEGAgeDataSetItem(data_dir=self.data_dir, data_ids=self.valid_set_ids, cust_graph=self.cust_graph)
        self.test_set = EEGAgeDataSetItem(data_dir=self.data_dir, data_ids=self.test_set_ids, cust_graph=self.cust_graph)

    def getBadSamples(self):
        bad_samples = []
        for bad_subject_id in Config.BAD_SUBJECT_IDS:
            for i in range(self.sample_split):
                cur_bad_sample_id = (bad_subject_id - 1) * self.sample_split + i + 1
                if cur_bad_sample_id <= self.n_samples:
                    bad_samples.append(
                            cur_bad_sample_id
                    )
        return bad_samples

    def getValidSamples(self):
        valid_samples = []
        for i in range(self.n_samples):
            cur_sample_id = i + 1
            if cur_sample_id not in self.bad_samples:
                valid_samples.append(cur_sample_id)
        return valid_samples

    def train_valid_split(self, n_train_valid):
        n_samples_per_fold = int(n_train_valid / self.folds)
        steps = [n_samples_per_fold * i for i in range(self.folds)]
        train, valid = [], []
        for i in range(len(steps) - 1):
            curFold = [x for x in range(steps[i], steps[i + 1])]
            if self.valid_k == i:
                valid += curFold
            else:
                train += curFold
        if self.valid_k == self.folds - 1 or self.valid_k == -1:
            valid += [x for x in range(steps[-1], n_train_valid)]
        else:
            train += [x for x in range(steps[-1], n_train_valid)]

        return train, valid


def evalSamplingSpeed(ds, batch_size, shuffle, tag, num_workers=4):
    """
    Test the sampling functionality & efficiency
    """
    dataloader = GraphDataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    time0 = time.time()

    for i, batch in enumerate(dataloader):
        features, graph, label = batch['inputs']['features'], batch['inputs']['graph'], batch['target']
        sys.stderr.write("\r{} Set - Batch No. {}/{} with time used(s): {}".format(tag, i + 1, len(dataloader), time.time() - time0))
        sys.stderr.flush()

    sys.stderr.write("\n")


if __name__ == '__main__':
    # dataset = EEGAgeDataSet(data_dir=Config.DATA_DIR_DEFAULT, n_samples=Config.NUM_SAMPLES, folds=5, valid_k=-1)
    dataset = EEGAgeDataSet(data_dir=Config.DATA_DIR_DEFAULT, n_samples=40, folds=5, valid_k=-1)
    evalSamplingSpeed(dataset.train_set, batch_size=5, shuffle=True, tag='Training', num_workers=4)
    evalSamplingSpeed(dataset.valid_set, batch_size=3, shuffle=False, tag='Validation', num_workers=4)
    evalSamplingSpeed(dataset.test_set, batch_size=3, shuffle=False, tag='Test', num_workers=4)
