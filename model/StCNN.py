import time

import torch
import torch.nn as nn

import Config


class StCNN(nn.Module):
    def __init__(self, feat_dim, hidden_dim, stride):
        super(StCNN, self).__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.stride = stride

        self.secGrabber = nn.Conv1d(in_channels=self.feat_dim, out_channels=self.hidden_dim, kernel_size=Config.EEG_FREQUENCY, stride=self.stride)
        self.bn = nn.BatchNorm1d(num_features=self.hidden_dim)

    def forward(self, feat: torch.Tensor):
        # feat is taken from graph, so N x T x 1 --> N x 1 x T
        # Then Conv1d and BatchNorm1d output N x h x T' --> N x T' x h
        inputs_sG = torch.transpose(feat, -2, -1)

        # Grab seconds
        out_sG = self.secGrabber(inputs_sG)
        del inputs_sG

        out_bn = self.bn(out_sG)
        del out_sG

        out_stCNN = torch.transpose(out_bn, -2, -1)
        del out_bn

        return out_stCNN


if __name__ == '__main__':
    # Before testing, remove dot ('.') in the import specification
    pack = torch.load('../data/EEG_age_data/1.pt')
    features = pack['V']

    stCNN = StCNN(hidden_dim=Config.HIDDEN_DIM_DEFAULT, stride=Config.STCNN_STRIDE)

    time0 = time.time()
    out = stCNN(features)
    print(out.shape, time.time() - time0, 'sec')
