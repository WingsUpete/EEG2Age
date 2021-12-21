import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import dgl
sys.stderr.close()
sys.stderr = stderr

import Config


class FeedForward(nn.Module):
    def __init__(self, num_channels=Config.NUM_NODES, num_timestamps=int(Config.TOTAL_TIMESTAMPS / Config.SAMPLE_SPLIT_DEFAULT)):
        super(FeedForward, self).__init__()
        self.node_dim = num_channels
        self.time_dim = num_timestamps

        self.linear_n = nn.Linear(in_features=self.node_dim, out_features=1, bias=True)
        self.linear_t = nn.Linear(in_features=self.time_dim, out_features=1, bias=True)

    def reset_parameters(self):
        """ Reinitialize learnable parameters. """
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear_n.weight, gain=gain)
        nn.init.xavier_normal_(self.linear_t.weight, gain=gain)

    def forward(self, inputs):
        x = inputs['features'].reshape(-1, self.node_dim, self.time_dim)
        x = torch.transpose(x, -2, -1)
        h = self.linear_n(x)
        z = self.linear_t(h.reshape(-1, self.time_dim))
        z = F.relu(z)
        del x
        del h
        return z


if __name__ == '__main__':
    pack = torch.load('../data/EEG_age_data/1.pt')
    features = pack['V']
    (graph,), _ = dgl.load_graphs('../data/EEG_age_data/graph.dgl')
    graph.ndata['v'] = features
    ins = {
        'features': features,
        'graph': graph
    }

    dense = FeedForward(num_channels=Config.NUM_NODES, num_timestamps=int(Config.TOTAL_TIMESTAMPS / Config.SAMPLE_SPLIT_DEFAULT))
    time0 = time.time()
    out = dense(ins)
    print(out, time.time() - time0, 'sec')
