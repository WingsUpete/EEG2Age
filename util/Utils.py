import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch


def path2FileNameWithoutExt(path):
    """
    get file name without extension from path
    :param path: file path
    :return: file name without extension
    """
    return os.path.splitext(os.path.basename(path))[0]


def MAE(y_pred: torch.Tensor, y_true: torch.Tensor):
    return torch.mean(torch.abs(y_true - y_pred))


def RMSE(y_pred: torch.Tensor, y_true: torch.Tensor):
    # return torch.sqrt(torch.mean(torch.pow(y_true - y_pred, 2)))  # sqrt should be outside
    return torch.mean(torch.pow(y_true - y_pred, 2))


ADDITIVE = 1e-10


def MAPE(y_pred: torch.Tensor, y_true: torch.Tensor):
    return torch.mean(torch.abs((y_true - y_pred) / (y_true + ADDITIVE)))


METRICS_FUNCTION_MAP = {
    'MAE': MAE,
    'RMSE': RMSE,
    'MAPE': MAPE
}


def constructMetricsStorage():
    metrics_map = {}
    for metrics in METRICS_FUNCTION_MAP:
        metrics_map[metrics] = 0
    return metrics_map


def aggMetricsWithMap(metrics_map, res, target):
    for metrics in metrics_map:
        metrics_map[metrics] += METRICS_FUNCTION_MAP[metrics](res, target).item()

    return metrics_map


def wrapMetricsWithMap(metrics_map, nBatch):
    for metrics in metrics_map:
        metrics_map[metrics] /= nBatch
        if metrics == 'RMSE':
            metrics_map[metrics] = torch.sqrt(metrics_map[metrics])

    return metrics_map


def metricsMap2Str(metrics_map):
    output = ['%s = %.4f' % (metrics, metrics_map[metrics]) for metrics in metrics_map]
    output = ', '.join(output)
    return output


def trainLog2LossCurve(logfn='train.log'):
    if not os.path.isfile(logfn):
        print('{} is not a valid file.'.format(logfn))
        exit(-1)

    x_epoch = []
    y_loss_train = []
    train_time_list = []

    print('Analyzing log file: {}'.format(logfn))
    f = open(logfn, 'r')
    lines = f.readlines()
    for line in lines:
        if not line.startswith('Training Round'):
            continue
        items = line.strip().split(sep=' ')

        epoch = int(items[2][:-1])
        x_epoch.append(epoch)

        loss = float(items[5][:-1])
        y_loss_train.append(loss)

        train_time = float(items[10][1:])
        train_time_list.append(train_time)

    # Count average TTpS
    avgTTpS = sum(train_time_list) / len(train_time_list)
    print('Average TTpS: %.4f sec' % avgTTpS)

    # Plot training loss curve
    print('Plotting loss curve.')
    plt.plot(x_epoch, y_loss_train, c='purple', label='Train Loss', alpha=0.8)
    plt.title('Epoch - Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    # plt.show()
    figpath = '{}.png'.format(path2FileNameWithoutExt(logfn))
    plt.savefig(figpath)
    print('Loss curve saved to {}'.format(figpath))

    print('All analysis tasks finished.')


# by RoshanRane in https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
def plot_grad_flow(named_parameters):
    """
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing/exploding problems.
    Usage: Plug this function in Trainer class after loss.backwards() as "plot_grad_flow(model.named_parameters())" to
        visualize the gradient flow.
    """
    avg_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and (p.grad is not None) and ('bias' not in n):
            layers.append(n)
            avg_grads.append(p.grad.cpu().abs().mean())
            max_grads.append(p.grad.cpu().abs().max())

    plt.figure(figsize=(7, 20))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color='c')
    plt.bar(np.arange(len(avg_grads)), avg_grads, alpha=0.1, lw=1, color='b')
    plt.hlines(0, 0, len(avg_grads)+1, lw=2, color='k')
    plt.xticks(range(0, len(avg_grads), 1), layers, rotation='vertical')
    plt.xlim(left=0, right=len(avg_grads))
    plt.ylim(bottom=-0.001, top=0.5)   # zoom in on the lower gradient regions
    plt.xlabel('Layers')
    plt.ylabel('Average Gradient')
    plt.title('Gradient flow')
    plt.grid(True)
    plt.legend([Line2D([0], [0], color='c', lw=4),
                Line2D([0], [0], color='b', lw=4),
                Line2D([0], [0], color='k', lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.tight_layout()
    plt.ion()
    plt.show()
    plt.pause(5)
    plt.ioff()


if __name__ == '__main__':
    trainLog2LossCurve('log/xx.log')
