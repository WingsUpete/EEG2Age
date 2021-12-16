import os
import sys
import multiprocessing
import argparse

import pandas as pd
import torch
from tqdm import tqdm

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import dgl
sys.stderr.close()
sys.stderr = stderr

DATA_SCALE = 1e6
PPC_THRESHOLD = 0.7
DATA_DIR_DEFAULT = '../data/EEG_age_data_raw'
OUT_DIR_DEFAULT = 'data/'
NUM_WORKERS_DEFAULT = 4
SAMPLE_SPLIT_DEFAULT = 4


def path2FileNameWithoutExt(path):
    """
    get file name without extension from path
    :param path: file path
    :return: file name without extension
    """
    return os.path.splitext(os.path.basename(path))[0]


def pushGraphEdge(gSrc: list, gDst: list, wList, src, dst, weight):
    gSrc.append(src)
    gDst.append(dst)
    if wList is not None and weight is not None:
        wList.append(weight)
        return gSrc, gDst, wList
    else:
        return gSrc, gDst


def matOD2G(mat, oList: list, dList: list, nGNodes):
    # pre weights
    matSum = torch.sum(mat, dim=0)
    for nj in range(nGNodes):
        if matSum[nj] == 0:
            continue
        for ni in range(nGNodes):
            mat[ni][nj] /= matSum[nj]

    # Transform node data mat to edge data edges
    edges = []
    for i in range(len(oList)):
        edges.append([mat[oList[i]][dList[i]]])

    # Create DGL Graph
    graph = dgl.graph((oList, dList), num_nodes=nGNodes)
    graph.edata['pre_w'] = torch.Tensor(edges).reshape(-1, 1, 1)

    return graph


def procData(data_path, out_dir, scale=DATA_SCALE, sample_split=4):
    # Read and scale data
    df = pd.read_csv(data_path)
    ts_df = torch.tensor(df.values).float()
    ts_df *= scale

    # Calculate Pearson Correlation Coefficient
    P = torch.corrcoef(torch.transpose(ts_df, 0, 1))

    # Split
    numT, numNodes = len(ts_df), len(ts_df[0])
    splitT = numT / sample_split
    if int(splitT) != splitT:
        sys.stderr.write('Warning: data T = %d not divisible by %d\n' % (numT, sample_split))
        exit(-7)
    splitT = int(splitT)
    samples = [torch.transpose(ts_df[i * splitT: (i + 1) * splitT], 0, 1).reshape(numNodes, splitT, 1)
               for i in range(sample_split)]

    # Save samples <(Subject_ID - 1) * N_Sample + Sample_ID + 1>.pt: {'V': numNodes * splitT * 1, 'target': 1 * 1}
    _, subId, target = path2FileNameWithoutExt(data_path).split('_')
    subId, target = float(subId), float(target)
    for i in range(len(samples)):
        out_fn = '%.f.pt' % (
            (int(subId) - 1) * sample_split + i + 1
        )
        out_path = os.path.join(out_dir, out_fn)
        outPack = {
            'V': samples[i],
            'target': torch.Tensor([[target]])
        }
        torch.save(outPack, out_path)

    return P


class GraphBuilder:
    """
    Scan through all csv files to build a graph structure for GAT.

    > 1. All the data values will be scaled up with a factor of 1e6.

    > 2. The edges are formed by calculating Pearson Correlation Coefficient (if over 0.7, form an edge). Each sample
    outputs a PCC matrix P. All Ps from the samples will be averaged and used to perform edge construction.
    Pre-weights of edges are calculated according to Ps.

    > 3. For each sample, the projected input features will be applied to each channel (node on the graph).
    All samples use the same graph structure (built from all samples).
    """
    def __init__(self, data_dir, out_dir, sample_split=SAMPLE_SPLIT_DEFAULT, num_workers=NUM_WORKERS_DEFAULT):
        if not os.path.exists(data_dir):
            sys.stderr.write('[GraphBuilder.init] %s does not exist!\n' % data_dir)
            exit(-23)

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        self.data_dir = data_dir
        self.out_dir = out_dir

        self.sample_split = sample_split
        self.num_workers = num_workers

        self.files = os.listdir(self.data_dir)
        self.procAll()

    def procAll(self):
        # Async spltting samples
        print('> Splitting samples')
        pool = multiprocessing.Pool(processes=self.num_workers)
        jobs = []
        for file in self.files:
            file_path = os.path.join(self.data_dir, file)
            jobs.append(pool.apply_async(procData, args=(file_path, self.out_dir, self.sample_split)))
        pool.close()

        Ps = []
        for job in tqdm(jobs):
            Ps.append(job.get())

        pool.join()
        print('> Splitting complete')

        # Construct a DGLGraph with pre-weights set
        print('> Constructing the graph according to PPC')
        P = sum(Ps) / len(Ps)
        P[P < PPC_THRESHOLD] = 0
        sL, dL = [], []
        num_nodes = P.shape[0]
        for mi in range(num_nodes):
            for mj in range(num_nodes):
                if P[mi][mj] != 0:  # PPC over threshold, they should form an edge
                    # Add once mi->mj is fine, since P is symmetric, mj->mi will be added when P[mj][mi] is considered
                    sL, dL = pushGraphEdge(sL, dL, None, mi, mj, None)
        g = matOD2G(mat=P, oList=sL, dList=dL, nGNodes=num_nodes)
        outGPath = os.path.join(self.out_dir, 'graph.dgl')
        dgl.save_graphs(outGPath, g)
        print('> Generated graph saved to %s' % outGPath)


if __name__ == '__main__':
    """
        Usage Example:
        python Trainer.py -dr data/xxx -c 4 -m train -net FeedForward
        python Trainer.py -dr data/xxx -c 4 -m eval -e model_save/xx.pth
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data', type=str, default=DATA_DIR_DEFAULT, help='Specify the data folder, default = {}'.format(DATA_DIR_DEFAULT))
    parser.add_argument('-c', '--cores', type=int, default=NUM_WORKERS_DEFAULT, help='Specify how many CPU cores to use, default = {}'.format(NUM_WORKERS_DEFAULT))
    parser.add_argument('-s', '--split', type=int, default=SAMPLE_SPLIT_DEFAULT, help='Specify how many sub-samples each sample is split into, default = {}'.format(SAMPLE_SPLIT_DEFAULT))
    parser.add_argument('-o', '--out', type=str, default=OUT_DIR_DEFAULT, help='Specify the output folder, default = {}'.format(OUT_DIR_DEFAULT))

    FLAGS, unparsed = parser.parse_known_args()

    gb = GraphBuilder(data_dir=FLAGS.data, out_dir=FLAGS.out, sample_split=FLAGS.split, num_workers=FLAGS.cores)
