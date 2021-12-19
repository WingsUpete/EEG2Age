import os
import sys
import time

import torch
import torch.nn as nn

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import dgl
sys.stderr.close()
sys.stderr = stderr

from .StCNN import StCNN
from .SpatAttLayer import SpatAttLayer
from .TempLayer import TempLayer
from .TranLayer import TranLayer


class BrainAgePredictionModel(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_nodes, num_heads):
        super(BrainAgePredictionModel, self).__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.num_heads = num_heads

        self.stC_embed_dim = self.hidden_dim            # Embedding dimension after short-term temporal feature extraction
        self.spat_embed_dim = 2 * self.stC_embed_dim    # Embedding dimension after spatial feature extraction
        self.temp_embed_dim = self.spat_embed_dim       # Embedding dimension after temporal feature extraction

        # Short-term Temporal Layer
        self.stCNN = StCNN(hidden_dim=self.hidden_dim)

        # Spatial Attention Layer
        self.spatAttLayer = SpatAttLayer(feat_dim=self.stC_embed_dim, hidden_dim=self.stC_embed_dim, num_nodes=self.num_nodes, num_heads=self.num_heads, gate=True, merge='mean')

        # Temporal Attention Layer
        self.tempAttLayer = TempLayer(embed_dim=self.spat_embed_dim, num_nodes=self.num_nodes)

        # Transfer Attention Layer
        self.tranAttLayer = TranLayer(embed_dim=self.temp_embed_dim, num_nodes=self.num_nodes)

    def forward(self, inputs: dict):
        g: dgl.DGLGraph = inputs['graph']

        # Short-term Temporal
        feat = g.ndata['v']
        convFeat = self.stCNN(feat)
        g.ndata['v'] = convFeat

        # Spatial
        spatFeat = self.spatAttLayer(g)

        # Temporal
        spatTempFeat = self.tempAttLayer(spatFeat)
        del spatFeat

        # Transfer
        pred = self.tranAttLayer(spatTempFeat)
        del spatTempFeat

        return pred * 50


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

    bapm = BrainAgePredictionModel(feat_dim=1, hidden_dim=2, num_nodes=features.shape[-3], num_heads=3)
    time0 = time.time()
    out = bapm(ins)
    print(out, time.time() - time0, 'sec')
