import time

import torch
import torch.nn as nn

import Config


class StCNN(nn.Module):
    def __init__(self, hidden_dim, num_nodes):
        super(StCNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes

        self.secGrabber = nn.Conv1d(in_channels=1, out_channels=self.hidden_dim,
                                    kernel_size=Config.EEG_FREQUENCY, stride=Config.EEG_FREQUENCY)
        self.bn = nn.BatchNorm1d(num_features=self.hidden_dim)

    def forward(self, feat: torch.Tensor):
        inputs_sG = torch.transpose(feat.reshape(-1, feat.shape[-2], 1), -2, -1)

        # Grab seconds
        out_sG = self.secGrabber(inputs_sG)
        del inputs_sG

        out_bn = self.bn(out_sG)
        del out_sG

        out_stCNN = torch.transpose(out_bn, -2, -1)
        out_stCNN = out_stCNN.reshape(-1, self.num_nodes, out_stCNN.shape[-2], self.hidden_dim)
        del out_bn

        return out_stCNN


if __name__ == '__main__':
    # Before testing, remove dot ('.') in the import specification
    pack = torch.load('../data/EEG_age_data/1.pt')
    features = pack['V']

    stCNN = StCNN(hidden_dim=Config.HIDDEN_DIM_DEFAULT, num_nodes=Config.NUM_NODES)

    time0 = time.time()
    out = stCNN(features)
    print(out.shape, time.time() - time0, 'sec')
