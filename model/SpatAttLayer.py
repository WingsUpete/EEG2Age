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

from .PwGaANLayer import MultiHeadPwGaANLayer


class SpatAttLayer(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_nodes, num_heads, gate=False, merge='mean'):
        super(SpatAttLayer, self).__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.num_heads = num_heads
        self.gate = gate
        self.merge = merge

        self.GaANBlk = MultiHeadPwGaANLayer(self.feat_dim, self.hidden_dim, self.num_nodes, self.num_heads, merge=self.merge, gate=self.gate)
        self.proj_fc = nn.Linear(self.feat_dim, self.hidden_dim, bias=False)

        # BatchNorm
        self.bn = nn.BatchNorm2d(num_features=self.hidden_dim * 2)
        if self.merge == 'mean':
            self.bn = nn.BatchNorm2d(num_features=self.hidden_dim * 2)
        elif self.merge == 'cat':
            self.bn = nn.BatchNorm2d(num_features=self.hidden_dim * (self.num_heads + 1))

        self.reset_parameters()

    def reset_parameters(self):
        """ Reinitialize learnable parameters. """
        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_normal_(self.proj_fc.weight, gain=gain)

    def forward(self, g: dgl.DGLGraph):
        feat = g.ndata['v']
        feat = F.dropout(feat, 0.1)
        g.ndata['v'] = feat

        proj_feat = self.proj_fc(feat)
        del feat

        g.ndata['proj_z'] = proj_feat

        out_proj_feat = proj_feat.reshape(g.batch_size, int(g.num_nodes() / g.batch_size), -1, self.hidden_dim)
        del proj_feat

        hg = self.GaANBlk(g)

        h = torch.cat([out_proj_feat, hg], dim=-1)
        del out_proj_feat
        del hg

        normH = self.bn(torch.transpose(h, -3, -1))
        reshapedH = torch.transpose(normH, -3, -1)
        del h
        del normH

        return reshapedH


if __name__ == '__main__':
    # Before testing, remove dot ('.') in the import specification
    pack = torch.load('../data/EEG_age_data/1.pt')
    features = pack['V']
    (graph,), _ = dgl.load_graphs('../data/EEG_age_data/graph.dgl')
    graph.ndata['v'] = features

    spat = SpatAttLayer(feat_dim=1, hidden_dim=1, num_nodes=features.shape[-3], num_heads=3, gate=True, merge='mean')

    time0 = time.time()
    out = spat(graph)
    print(out, time.time() - time0, 'sec')
