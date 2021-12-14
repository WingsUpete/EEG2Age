import numpy as np
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .PwGaANLayer import MultiHeadPwGaANLayer


class SpatAttLayer(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_heads, gate=False, merge='mean'):
        super(SpatAttLayer, self).__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.gate = gate
        self.merge = merge

        self.GaANBlk = MultiHeadPwGaANLayer(self.feat_dim, self.hidden_dim, self.num_heads, gate=self.gate, merge=self.merge)
        self.proj_fc = nn.Linear(self.feat_dim, self.hidden_dim, bias=False)

        # BatchNorm
        self.bn = nn.BatchNorm1d(num_features=self.hidden_dim * 2)
        if self.merge == 'mean':
            self.bn = nn.BatchNorm1d(num_features=self.hidden_dim * 2)
        elif self.merge == 'cat':
            self.bn = nn.BatchNorm1d(num_features=self.hidden_dim * (self.num_heads + 1))

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

        out_proj_feat = proj_feat.reshape(g.batch_size, -1, self.hidden_dim)
        del proj_feat

        hg = self.GaANBlk(g)

        h = torch.cat([out_proj_feat, hg], dim=-1)
        del out_proj_feat
        del hg

        normH = self.bn(torch.transpose(h, -2, -1))
        reshapedH = torch.transpose(normH, -2, -1)
        del h
        del normH

        return reshapedH

