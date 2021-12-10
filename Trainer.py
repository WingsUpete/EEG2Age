import sys
import os
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.autograd.profiler as profiler

from util import Logger, plot_grad_flow, MAE
from EEGAgeDataSet import EEGAgeDataSet
from model import FeedForward

import Config
if Config.CHECK_GRADS:
    torch.autograd.set_detect_anomaly(True)


def train(lr=Config.LEARNING_RATE_DEFAULT, bs=Config.BATCH_SIZE_DEFAULT, ep=Config.MAX_EPOCHS_DEFAULT,
          eval_freq=Config.EVAL_FREQ_DEFAULT, opt=Config.OPTIMIZER_DEFAULT, num_workers=Config.WORKERS_DEFAULT,
          use_gpu=True, gpu_id=Config.GPU_ID_DEFAULT, data_dir=Config.DATA_DIR_DEFAULT, logr=None,
          model=Config.NETWORK_DEFAULT, model_save_dir=Config.MODEL_SAVE_DIR_DEFAULT,
          loss_function=Config.LOSS_FUNC_DEFAULT,
          feat_dim=Config.FEAT_DIM_DEFAULT, hidden_dim=Config.HIDDEN_DIM_DEFAULT,
          folds=Config.FOLDS_DEFAULT, kid=Config.VALID_K_DEFAULT):
    # CUDA if possible
    device = torch.device('cuda:%d' % gpu_id if (use_gpu and torch.cuda.is_available()) else 'cpu')
    logr.log('> device: {}\n'.format(device))

    # Load DataSet
    logr.log('> Loading DataSet from {}\n'.format(data_dir))
    dataset = EEGAgeDataSet(data_dir, folds=folds, valid_k=kid)
    trainloader = DataLoader(dataset.train_set, batch_size=bs, shuffle=True, num_workers=num_workers)
    validloader = DataLoader(dataset.valid_set, batch_size=bs, shuffle=False, num_workers=num_workers)
    logr.log('> Training batches: {}, Validation batches: {}\n'.format(len(trainloader), len(validloader)))

    # Initialize the Model
    net = FeedForward(in_dim=feat_dim, hidden_dim=hidden_dim)
    logr.log('> Initializing the Training Model: {}\n'.format(model))
    if model == 'FeedForward':
        net = FeedForward(in_dim=feat_dim, hidden_dim=hidden_dim)
    logr.log('> Model Structure:\n{}\n'.format(net))
    if device:
        net.to(device)
        logr.log('> Model sent to {}\n'.format(device))

    # Loss Function
    logr.log('> Using {} as the Loss Function.\n'.format(loss_function))
    criterion = nn.SmoothL1Loss(beta=Config.SMOOTH_L1_LOSS_BETA_DEFAULT)
    if loss_function == 'SmoothL1Loss':
        criterion = nn.SmoothL1Loss(beta=Config.SMOOTH_L1_LOSS_BETA_DEFAULT)
    elif loss_function == 'MSELoss':
        criterion = nn.MSELoss()

    # Optimizer
    logr.log('> Constructing the Optimizer: {}\n'.format(opt))
    optimizer = torch.optim.Adam(net.parameters(), lr, weight_decay=Config.WEIGHT_DECAY_DEFAULT)
    if opt == 'ADAM':
        optimizer = torch.optim.Adam(net.parameters(), lr, weight_decay=Config.WEIGHT_DECAY_DEFAULT)  # Adam + L2 Norm

    # Model Saving Directory
    if not os.path.isdir(model_save_dir):
        os.mkdir(model_save_dir)

    # Summarize Info
    logr.log('\nlearning_rate = {}, epochs = {}, num_workers = {}\n'.format(lr, ep, num_workers))
    logr.log('eval_freq = {}, batch_size = {}, optimizer = {}\n'.format(eval_freq, bs, opt))
    logr.log('folds = {}, valid_fold_id = {}\n'.format(folds, kid))

    # Start Training
    logr.log('\nStart Training!\n')
    logr.log('------------------------------------------------------------------------\n')

    min_eval_loss = float('inf')
    for epoch_i in range(ep):
        # train one round
        net.train()
        train_loss = 0
        train_mae = 0
        time_start_train = time.time()
        for i, batch in enumerate(trainloader):
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            features, target = batch['features'], batch['target']
            if device:
                features = features.to(device)
                target = target.to(device)

            # Avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=Config.MAX_NORM_DEFAULT)

            optimizer.zero_grad()

            if Config.PROFILE:
                with profiler.profile(profile_memory=True, use_cuda=True) as prof:
                    with profiler.record_function('model_inference'):
                        res = net(batch)
                logr.log(prof.key_averages().table(sort_by="cuda_time_total"))
                exit(100)

            res = net(batch)

            loss = criterion(res, target)
            loss.backward()

            if Config.CHECK_GRADS:
                plot_grad_flow(net.named_parameters())

            optimizer.step()

            with torch.no_grad():
                train_loss += loss.item()
                train_mae += MAE(res, target)

            if Config.TRAIN_JUST_ONE_BATCH:     # DEBUG
                if i == 0:
                    break

        train_loss /= len(trainloader)
        train_mae /= len(trainloader)
        time_end_train = time.time()
        total_train_time = (time_end_train - time_start_train)
        train_time_per_sample = total_train_time / len(dataset.train_set)
        logr.log('Training Round %d: loss = %.6f, time_cost = %.4f sec (%.4f sec per sample), MAE = %.4f\n' %
                 (epoch_i + 1, train_loss, total_train_time, train_time_per_sample, train_mae))

        # eval_freq: Evaluate on validation set
        if (epoch_i + 1) % eval_freq == 0:
            net.eval()
            val_mae = 0
            val_loss = 0
            with torch.no_grad():
                for j, val_batch in enumerate(validloader):
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

                    val_features, val_target = val_batch['features'], val_batch['target']
                    if device:
                        val_features = val_features.to(device)
                        val_target = val_target.to(device)

                    val_res = net(val_features)

                    val_loss = criterion(val_res, val_target)

                    val_loss += val_loss.item()
                    val_mae += MAE(val_res, val_target)

                val_loss /= len(validloader)
                val_mae /= len(validloader)
                logr.log('!!! Validation: loss = %.6f, MAE = %.4f\n' % (val_loss, val_mae))

                # Save model if we have better validation results
                if val_loss < min_eval_loss:
                    min_eval_loss = val_loss
                    model_path = os.path.join(model_save_dir, '{}.pth'.format(logr.time_tag))
                    torch.save(net, model_path)
                    logr.log('Model: {} has been saved since it achieves smaller loss.\n'.format(model_path))

        if Config.TRAIN_JUST_ONE_ROUND:
            if epoch_i == 0:    # DEBUG
                break

    # End Training
    logr.log('> Training finished.\n')


def evalMetrics(dataloader: DataLoader, device: torch.device, net):
    mae = 0
    for j, batch in enumerate(dataloader):
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        features, target = batch['features'], batch['target']
        if device:
            features = features.to(device)
            target = target.to(device)

        res = net(features)
        mae += MAE(res, target)
    mae /= len(dataloader)
    return mae


def evaluate(model_path, bs=Config.BATCH_SIZE_DEFAULT, num_workers=Config.WORKERS_DEFAULT,
             use_gpu=True, gpu_id=Config.GPU_ID_DEFAULT,
             data_dir=Config.DATA_DIR_DEFAULT, logr=None,
             folds=Config.FOLDS_DEFAULT, kid=Config.VALID_K_DEFAULT):
    # CUDA if needed
    device = torch.device('cuda:%d' % gpu_id if (use_gpu and torch.cuda.is_available()) else 'cpu')
    logr.log('> device: {}\n'.format(device))

    # Load Model
    logr.log('> Loading {}\n'.format(model_path))
    net = torch.load(model_path)
    logr.log('> Model Structure:\n{}\n'.format(net))
    if device:
        net.to(device)
        logr.log('> Model sent to {}\n'.format(device))

    # Load DataSet
    logr.log('> Loading DataSet from {}\n'.format(data_dir))
    dataset = EEGAgeDataSet(data_dir, folds=folds, valid_k=kid)
    validloader = DataLoader(dataset.valid_set, batch_size=bs, shuffle=False, num_workers=num_workers)
    testloader = DataLoader(dataset.test_set, batch_size=bs, shuffle=False, num_workers=num_workers)
    logr.log('> Validation batches: {}, Test batches: {}\n'.format(len(validloader), len(testloader)))

    # Evaluate
    net.eval()

    # - Validation
    val_mae = evalMetrics(validloader, device, net)
    logr.log('Validation MAE = %.4f\n' % val_mae)

    # - Test
    test_mae = evalMetrics(testloader, device, net)
    logr.log('Test MAE = %.4f\n' % test_mae)

    # End Evaluation
    logr.log('> Evaluation finished.\n')


if __name__ == '__main__':
    """
        Usage Example:
        python Trainer.py -dr data/xxx -c 4 -m train -net FeedForward
        python Trainer.py -dr data/xxx -c 4 -m eval -e model_save/xx.pth
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-lr', '--learning_rate', type=float, default=Config.LEARNING_RATE_DEFAULT, help='Learning rate, default = {}'.format(Config.LEARNING_RATE_DEFAULT))
    parser.add_argument('-me', '--max_epochs', type=int, default=Config.MAX_EPOCHS_DEFAULT, help='Number of epochs to run the trainer, default = {}'.format(Config.MAX_EPOCHS_DEFAULT))
    parser.add_argument('-ef', '--eval_freq', type=int, default=Config.EVAL_FREQ_DEFAULT, help='Frequency of evaluation on the validation set, default = {}'.format(Config.EVAL_FREQ_DEFAULT))
    parser.add_argument('-bs', '--batch_size', type=int, default=Config.BATCH_SIZE_DEFAULT, help='Size of a minibatch, default = {}'.format(Config.BATCH_SIZE_DEFAULT))
    parser.add_argument('-opt', '--optimizer', type=str, default=Config.OPTIMIZER_DEFAULT, help='Optimizer to be used, default = {}'.format(Config.OPTIMIZER_DEFAULT))
    parser.add_argument('-dr', '--data_dir', type=str, default=Config.DATA_DIR_DEFAULT, help='Root directory of the input data, default = {}'.format(Config.DATA_DIR_DEFAULT))
    parser.add_argument('-ld', '--log_dir', type=str, default=Config.LOG_DIR_DEFAULT, help='Specify where to create a log file. If log files are not wanted, value will be None'.format(Config.LOG_DIR_DEFAULT))
    parser.add_argument('-c', '--cores', type=int, default=Config.WORKERS_DEFAULT, help='number of workers (cores used), default = {}'.format(Config.WORKERS_DEFAULT))
    parser.add_argument('-gpu', '--gpu', type=int, default=Config.USE_GPU_DEFAULT, help='Specify whether to use GPU, default = {}'.format(Config.USE_GPU_DEFAULT))
    parser.add_argument('-gid', '--gpu_id', type=int, default=Config.GPU_ID_DEFAULT, help='Specify which GPU to use, default = {}'.format(Config.GPU_ID_DEFAULT))
    parser.add_argument('-net', '--network', type=str, default=Config.NETWORK_DEFAULT, help='Specify which model/network to use, default = {}'.format(Config.NETWORK_DEFAULT))
    parser.add_argument('-m', '--mode', type=str, default=Config.MODE_DEFAULT, help='Specify which mode the discriminator runs in (train, eval), default = {}'.format(Config.MODE_DEFAULT))
    parser.add_argument('-e', '--eval', type=str, default=Config.EVAL_DEFAULT, help='Specify the location of saved network to be loaded for evaluation, default = {}'.format(Config.EVAL_DEFAULT))
    parser.add_argument('-md', '--model_save_dir', type=str, default=Config.MODEL_SAVE_DIR_DEFAULT, help='Specify the location of network to be saved, default = {}'.format(Config.MODEL_SAVE_DIR_DEFAULT))
    parser.add_argument('-fd', '--feature_dim', type=int, default=Config.FEAT_DIM_DEFAULT, help='Specify the feature dimension, default = {}'.format(Config.FEAT_DIM_DEFAULT))
    parser.add_argument('-hd', '--hidden_dim', type=int, default=Config.HIDDEN_DIM_DEFAULT, help='Specify the hidden dimension, default = {}'.format(Config.HIDDEN_DIM_DEFAULT))
    parser.add_argument('-lf', '--loss_function', type=str, default=Config.LOSS_FUNC_DEFAULT, help='Specify which loss function to use, default = {}'.format(Config.LOSS_FUNC_DEFAULT))
    parser.add_argument('-f', '--folds', type=int, default=Config.FOLDS_DEFAULT, help='Number of folds, default = {}'.format(Config.FOLDS_DEFAULT))
    parser.add_argument('-k', '--k_id', type=int, default=Config.VALID_K_DEFAULT, help='Fold number k (index) used for validation, default = {}'.format(Config.VALID_K_DEFAULT))

    FLAGS, unparsed = parser.parse_known_args()

    # Starts a log file in the specified directory
    logger = Logger(activate=True, logging_folder=FLAGS.log_dir) if FLAGS.log_dir else Logger(activate=False)

    working_mode = FLAGS.mode
    if working_mode == 'train':
        train(lr=FLAGS.learning_rate, bs=FLAGS.batch_size, ep=FLAGS.max_epochs,
              eval_freq=FLAGS.eval_freq, opt=FLAGS.optimizer, num_workers=FLAGS.cores,
              use_gpu=(FLAGS.gpu == 1), gpu_id=FLAGS.gpu_id, data_dir=FLAGS.data_dir, logr=logger,
              model=FLAGS.network, model_save_dir=FLAGS.model_save_dir,
              loss_function=FLAGS.loss_function,
              feat_dim=FLAGS.feature_dim, hidden_dim=FLAGS.hidden_dim,
              folds=FLAGS.folds, kid=FLAGS.k_id)
        logger.close()
    elif working_mode == 'eval':
        eval_file = FLAGS.eval
        # Abnormal: file not found
        if (not eval_file) or (not os.path.isfile(eval_file)):
            sys.stderr.write('File for evaluation not found, please check!\n')
            logger.close()
            exit(-1)
        # Normal
        evaluate(eval_file, bs=FLAGS.batch_size, num_workers=FLAGS.cores,
                 use_gpu=(FLAGS.gpu == 1), gpu_id=FLAGS.gpu_id,
                 data_dir=FLAGS.data_dir, logr=logger,
                 folds=FLAGS.folds, kid=FLAGS.k_id)
        logger.close()
    else:
        sys.stderr.write('Please specify the working mode (train/eval)\n')
        logger.close()
        exit(-2)
