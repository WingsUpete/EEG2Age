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

from .StCNN import StCNN
from .SpatAttLayer import SpatAttLayer
from .TempLayer import TempLayer
from .TranLayer import TranLayer

import Config


class BrainAgePredictionModel(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_nodes, stCNN_stride, num_heads):
        super(BrainAgePredictionModel, self).__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.stCNN_stride = stCNN_stride
        self.num_heads = num_heads

        self.stC_embed_dim = self.hidden_dim            # Embedding dimension after short-term temporal feature extraction
        self.spat_embed_dim = 2 * self.stC_embed_dim    # Embedding dimension after spatial feature extraction
        self.temp_embed_dim = self.spat_embed_dim       # Embedding dimension after temporal feature extraction

        # Short-term Temporal Layer
        self.stCNN = StCNN(feat_dim=self.feat_dim, hidden_dim=self.hidden_dim, stride=self.stCNN_stride)

        # Spatial Attention Layer
        self.spatAttLayer = SpatAttLayer(feat_dim=self.stC_embed_dim, hidden_dim=self.stC_embed_dim, num_nodes=self.num_nodes, num_heads=self.num_heads, gate=True, merge='mean')

        # Temporal Layer
        self.tempLayer = TempLayer(embed_dim=self.spat_embed_dim, num_nodes=self.num_nodes)

        # Transfer Layer
        self.tranLayer = TranLayer(embed_dim=self.temp_embed_dim, num_nodes=self.num_nodes)

    def forward(self, inputs: dict):
        g: dgl.DGLGraph = inputs['graph']

        # Short-term Temporal
        feat = g.ndata['v']
        convFeat = self.stCNN(feat)
        g.ndata['v'] = convFeat

        # Spatial
        spatFeat = self.spatAttLayer(g)

        # Temporal
        spatTempFeat = self.tempLayer(spatFeat)
        del spatFeat

        # Transfer
        pred = self.tranLayer(spatTempFeat)
        del spatTempFeat

        return pred


# Ablation Experiment
class BAPM1(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_nodes, stCNN_stride, num_timestamps):
        super(BAPM1, self).__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.stCNN_stride = stCNN_stride
        self.num_timestamps = num_timestamps

        self.stC_embed_dim = self.hidden_dim  # Embedding dimension after short-term temporal feature extraction

        # Short-term Temporal Layer
        self.stCNN = StCNN(feat_dim=self.feat_dim, hidden_dim=self.hidden_dim, stride=self.stCNN_stride)

        # Calculate time dimension after short-term convolution
        self.stTDim = int(
            (self.num_timestamps - Config.EEG_FREQUENCY) / self.stCNN_stride + 1
        )

        self.linear_time = nn.Linear(in_features=self.stTDim, out_features=1, bias=True)

        # Transfer Layer
        self.tranLayer = TranLayer(embed_dim=self.stC_embed_dim, num_nodes=self.num_nodes)

    def reset_parameters(self):
        """ Reinitialize learnable parameters. """
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear_time.weight, gain=gain)

    def forward(self, inputs: dict):
        # Short-term Temporal
        feat = inputs['features']
        feat = feat.reshape(-1, feat.shape[-2], self.feat_dim)
        convFeat = self.stCNN(feat)
        convFeat = convFeat.reshape(-1, self.num_nodes, convFeat.shape[-2], self.hidden_dim)

        # Time: bs x N x T' x h --> bs x N x h x T' --> bs x N x h x 1 --> bs x N x h
        aggT = self.linear_time(torch.transpose(convFeat, -2, -1))
        aggT = F.relu(aggT)
        aggT = aggT.reshape(-1, self.num_nodes, self.stC_embed_dim)
        del convFeat

        # Transfer
        pred = self.tranLayer(aggT)
        del aggT

        return pred


class BAPM2(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_nodes, stCNN_stride, num_heads, num_timestamps):
        super(BAPM2, self).__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.stCNN_stride = stCNN_stride
        self.num_heads = num_heads
        self.num_timestamps = num_timestamps

        self.stC_embed_dim = self.hidden_dim            # Embedding dimension after short-term temporal feature extraction
        self.spat_embed_dim = 2 * self.stC_embed_dim    # Embedding dimension after spatial feature extraction

        # Short-term Temporal Layer
        self.stCNN = StCNN(feat_dim=self.feat_dim, hidden_dim=self.hidden_dim, stride=self.stCNN_stride)

        # Spatial Attention Layer
        self.spatAttLayer = SpatAttLayer(feat_dim=self.stC_embed_dim, hidden_dim=self.stC_embed_dim, num_nodes=self.num_nodes, num_heads=self.num_heads, gate=True, merge='mean')

        # Calculate time dimension after short-term convolution
        self.stTDim = int(
            (self.num_timestamps - Config.EEG_FREQUENCY) / self.stCNN_stride + 1
        )

        self.linear_time = nn.Linear(in_features=self.stTDim, out_features=1, bias=True)

        # Transfer Layer
        self.tranLayer = TranLayer(embed_dim=self.spat_embed_dim, num_nodes=self.num_nodes)

    def reset_parameters(self):
        """ Reinitialize learnable parameters. """
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear_time.weight, gain=gain)

    def forward(self, inputs: dict):
        g: dgl.DGLGraph = inputs['graph']

        # Short-term Temporal
        feat = g.ndata['v']
        convFeat = self.stCNN(feat)
        g.ndata['v'] = convFeat

        # Spatial
        spatFeat = self.spatAttLayer(g)

        # Time: bs x N x T' x h --> bs x N x h x T' --> bs x N x h x 1 --> bs x N x h
        aggT = self.linear_time(torch.transpose(spatFeat, -2, -1))
        aggT = F.relu(aggT)
        aggT = aggT.reshape(-1, self.num_nodes, self.spat_embed_dim)
        del spatFeat

        # Transfer
        pred = self.tranLayer(aggT)
        del aggT

        return pred


if __name__ == '__main__':
    # Before testing, remove dot ('.') in the import specification
    pack = torch.load('../data/EEG_age_data/1.pt')
    features = pack['V']
    (graph,), _ = dgl.load_graphs('../data/EEG_age_data/graph.dgl')
    graph.ndata['v'] = features
    ins = {
        'features': features,
        'graph': graph
    }

    bapm = BrainAgePredictionModel(feat_dim=1, hidden_dim=2, num_nodes=features.shape[-3], stCNN_stride=1024, num_heads=3)
    time0 = time.time()
    out = bapm(ins)
    print(out, time.time() - time0, 'sec')
