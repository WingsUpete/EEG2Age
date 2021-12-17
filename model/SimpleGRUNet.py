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


class GRUNet(nn.Module):
    def __init__(self, hidden_dim=1, num_nodes=Config.NUM_NODES):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes

        self.linear_n = nn.Linear(in_features=self.num_nodes, out_features=self.hidden_dim, bias=True)
        self.linear_o = nn.Linear(in_features=self.hidden_dim, out_features=1, bias=True)

        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim)

    def reset_parameters(self):
        """ Reinitialize learnable parameters. """
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear_n.weight, gain=gain)
        nn.init.xavier_normal_(self.linear_o.weight, gain=gain)

    def forward(self, inputs):
        # `features` is of shape: BS * NODE * TIME * 1
        # need to change to: TIME * BS * NODE --> TIME * BS * HIDDEN
        x = inputs['features']
        x = x.reshape(-1, self.num_nodes, x.shape[-2]).permute(2, 0, 1)

        # Projection
        z = self.linear_n(x)

        # GRU
        o, h = self.gru(z)
        del x
        del h

        # Dense
        res = self.linear_o(o[-1])
        res = F.relu(res)
        del o

        return res


if __name__ == '__main__':
    pack = torch.load('../data/EEG_age_data/1.pt')
    features = pack['V']
    (graph,), _ = dgl.load_graphs('../data/EEG_age_data/graph.dgl')
    graph.ndata['v'] = features
    ins = {
        'features': features,
        'graph': graph
    }

    gruNet = GRUNet(hidden_dim=1, num_nodes=Config.NUM_NODES)
    time0 = time.time()
    out = gruNet(ins)
    print(out, time.time() - time0, 'sec')
