import torch
import torch.nn as nn
import torch.nn.functional as F


class TranLayer(nn.Module):
    def __init__(self, embed_dim, num_nodes):
        super(TranLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_nodes = num_nodes

        self.linear_nodes = nn.Linear(in_features=self.num_nodes, out_features=1, bias=True)
        self.linear_embed = nn.Linear(in_features=self.embed_dim, out_features=1, bias=True)

    def reset_parameters(self):
        """ Reinitialize learnable parameters. """
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear_nodes.weight, gain=gain)
        nn.init.xavier_normal_(self.linear_embed.weight, gain=gain)

    def forward(self, embed_feat: torch.Tensor):
        # `embed_feat` is of shape: BS * NODE * FEAT
        # need to change to: BS * FEAT * NODE -> BS * FEAT -> BS * 1
        reshapedEmbedFeat = torch.transpose(embed_feat, -2, -1)
        aggNodes = self.linear_nodes(reshapedEmbedFeat)
        aggNodes = F.relu(aggNodes).reshape(-1, self.embed_dim)
        del reshapedEmbedFeat

        aggEmbed = self.linear_embed(aggNodes)
        aggEmbed = F.relu(aggEmbed)
        del aggNodes

        return aggEmbed
