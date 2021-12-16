import torch
import torch.nn as nn


class TempLayer(nn.Module):
    def __init__(self, embed_dim, num_nodes):
        super(TempLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_nodes = num_nodes

        self.gru_embed_dim = int(self.embed_dim * self.num_nodes)
        self.gru = nn.GRU(self.gru_embed_dim, self.gru_embed_dim)

        self.bn = nn.BatchNorm1d(num_features=self.embed_dim)

    def forward(self, embed_feat: torch.Tensor):
        # `embed_feat` is of shape: BS * NODE * TIME * FEAT
        # need to change to: TIME * BS * (NODE * FEAT)
        x = embed_feat.permute(2, 0, 1, 3)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        o, h = self.gru(x)
        del x
        del h

        # Transfer back
        o = o[-1].reshape(o.shape[1], self.num_nodes, self.embed_dim)

        # Batch Norm
        normO = self.bn(torch.transpose(o, -2, -1))
        reshapedO = torch.transpose(normO, -2, -1)
        del o
        del normO

        return reshapedO
