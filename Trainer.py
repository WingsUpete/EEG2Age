import argparse
import os
import sys
import time

import torch
import torch.autograd.profiler as profiler
import torch.nn as nn

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from dgl.dataloading import GraphDataLoader
sys.stderr.close()
sys.stderr = stderr

from util import Logger, plot_grad_flow, constructMetricsStorage, aggMetricsWithMap, wrapMetricsWithMap, metricsMap2Str
from EEGAgeDataSet import EEGAgeDataSet
from model import FeedForward, GRUNet, BrainAgePredictionModel, BAPM1, BAPM2

import Config
if Config.CHECK_GRADS:
    torch.autograd.set_detect_anomaly(True)


def batch2device(batch: dict, device: torch.device):
    batch['inputs']['features'] = batch['inputs']['features'].to(device)
    batch['inputs']['graph'] = batch['inputs']['graph'].to(device)
    batch['target'] = batch['target'].to(device)
    return batch


def train(lr=Config.LEARNING_RATE_DEFAULT, bs=Config.BATCH_SIZE_DEFAULT, ep=Config.MAX_EPOCHS_DEFAULT,
          eval_freq=Config.EVAL_FREQ_DEFAULT, opt=Config.OPTIMIZER_DEFAULT, num_workers=Config.WORKERS_DEFAULT,
          use_gpu=True, gpu_id=Config.GPU_ID_DEFAULT, logr=None,
          data_dir=Config.DATA_DIR_DEFAULT, n_data_samples=Config.NUM_SAMPLES_DEFAULT, cust_graph=False,
          model=Config.NETWORK_DEFAULT, model_save_dir=Config.MODEL_SAVE_DIR_DEFAULT,
          loss_function=Config.LOSS_FUNC_DEFAULT,
          feat_dim=Config.FEAT_DIM_DEFAULT, hidden_dim=Config.HIDDEN_DIM_DEFAULT,
          folds=Config.FOLDS_DEFAULT, kid=Config.VALID_K_DEFAULT,
          sample_split=Config.SAMPLE_SPLIT_DEFAULT, stCNN_stride=Config.STCNN_STRIDE_DEFAULT):
    # CUDA if possible
    device = torch.device('cuda:%d' % gpu_id if (use_gpu and torch.cuda.is_available()) else 'cpu')
    logr.log('> device: {}\n'.format(device))

    # Load DataSet
    logr.log('> Loading DataSet from %s, given %d samples%s\n' %
             (data_dir, n_data_samples, ', using a customized graph' if cust_graph else '')
             )
    dataset = EEGAgeDataSet(data_dir, n_samples=n_data_samples, sample_split=sample_split, cust_graph=cust_graph, folds=folds, valid_k=kid)
    trainloader = GraphDataLoader(dataset.train_set, batch_size=bs, shuffle=True, num_workers=num_workers)
    validloader = GraphDataLoader(dataset.valid_set, batch_size=bs, shuffle=False, num_workers=num_workers)
    logr.log('> Training batches: {}, Validation batches: {}\n'.format(len(trainloader), len(validloader)))

    # Initialize the Model
    net = BrainAgePredictionModel(feat_dim=feat_dim, hidden_dim=hidden_dim, num_nodes=Config.NUM_NODES, stCNN_stride=stCNN_stride, num_heads=Config.NUM_HEADS_DEFAULT)
    logr.log('> Initializing the Training Model: {}\n'.format(model))
    if model == 'FeedForward':
        num_timestamps = int(Config.TOTAL_TIMESTAMPS / sample_split)
        net = FeedForward(num_channels=Config.NUM_NODES, num_timestamps=num_timestamps)
    elif model == 'GRUNet':
        net = GRUNet(hidden_dim=hidden_dim, num_nodes=Config.NUM_NODES)
    elif model == 'BAPM':
        net = BrainAgePredictionModel(feat_dim=feat_dim, hidden_dim=hidden_dim, num_nodes=Config.NUM_NODES, stCNN_stride=stCNN_stride, num_heads=Config.NUM_HEADS_DEFAULT)
    elif model == 'BAPM1':
        num_timestamps = int(Config.TOTAL_TIMESTAMPS / sample_split)
        net = BAPM1(feat_dim=feat_dim, hidden_dim=hidden_dim, num_nodes=Config.NUM_NODES, stCNN_stride=stCNN_stride, num_timestamps=num_timestamps)
    elif model == 'BAPM2':
        num_timestamps = int(Config.TOTAL_TIMESTAMPS / sample_split)
        net = BAPM2(feat_dim=feat_dim, hidden_dim=hidden_dim, num_nodes=Config.NUM_NODES, stCNN_stride=stCNN_stride, num_heads=Config.NUM_HEADS_DEFAULT, num_timestamps=num_timestamps)
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
        train_metrics = constructMetricsStorage()
        time_start_train = time.time()
        for i, batch in enumerate(trainloader):
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            if device:
                batch = batch2device(batch, device)
            features, target = batch['inputs'], batch['target']

            # Avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=Config.MAX_NORM_DEFAULT)

            optimizer.zero_grad()

            if Config.PROFILE:
                with profiler.profile(profile_memory=True, use_cuda=True) as prof:
                    with profiler.record_function('model_inference'):
                        res = net(features)
                logr.log(prof.key_averages().table(sort_by="cuda_time_total"))
                exit(100)

            res = net(features)

            loss = criterion(res, target)
            loss.backward()

            if Config.CHECK_GRADS:
                plot_grad_flow(net.named_parameters())

            optimizer.step()

            with torch.no_grad():
                train_loss += loss.item()
                train_metrics = aggMetricsWithMap(train_metrics, res, target)
                del features
                del target
                del res

            if Config.TRAIN_JUST_ONE_BATCH:     # DEBUG
                if i == 0:
                    break

        train_loss /= len(trainloader)
        train_metrics = wrapMetricsWithMap(train_metrics, len(trainloader))
        time_end_train = time.time()
        total_train_time = (time_end_train - time_start_train)
        train_time_per_sample = total_train_time / len(dataset.train_set)
        logr.log('Training Round %d: loss = %.6f, time_cost = %.4f sec (%.4f sec per sample), %s\n' %
                 (epoch_i + 1, train_loss, total_train_time, train_time_per_sample, metricsMap2Str(train_metrics)))

        # eval_freq: Evaluate on validation set
        if (epoch_i + 1) % eval_freq == 0:
            net.eval()
            val_metrics = constructMetricsStorage()
            val_loss_total = 0
            with torch.no_grad():
                for j, val_batch in enumerate(validloader):
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

                    if device:
                        val_batch = batch2device(val_batch, device)
                    val_features, val_target = val_batch['inputs'], val_batch['target']

                    val_res = net(val_features)

                    val_loss = criterion(val_res, val_target)

                    val_loss_total += val_loss.item()
                    val_metrics = aggMetricsWithMap(val_metrics, val_res, val_target)
                    del val_features
                    del val_target
                    del val_res

                val_loss_total /= len(validloader)
                val_metrics = wrapMetricsWithMap(val_metrics, len(validloader))
                logr.log('!!! Validation: loss = %.6f, %s\n' % (val_loss_total, metricsMap2Str(val_metrics)))

                # Save model if we have better validation results
                if epoch_i >= 10 and val_loss_total < min_eval_loss:
                    min_eval_loss = val_loss_total
                    model_path = os.path.join(model_save_dir, '{}.pth'.format(logr.time_tag))
                    torch.save(net, model_path)
                    logr.log('Model: {} has been saved since it achieves smaller loss.\n'.format(model_path))

        if Config.TRAIN_JUST_ONE_ROUND:
            if epoch_i == 0:    # DEBUG
                break

    # End Training
    logr.log('> Training finished.\n')


def evalMetrics(dataloader: GraphDataLoader, device: torch.device, net):
    metrics = constructMetricsStorage()
    for j, batch in enumerate(dataloader):
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        if device:
            batch = batch2device(batch, device)
        features, target = batch['inputs'], batch['target']

        res = net(features)
        metrics = aggMetricsWithMap(metrics, res, target)
        del features
        del target
        del res

    metrics = wrapMetricsWithMap(metrics, len(dataloader))
    return metrics


def evaluate(model_path, bs=Config.BATCH_SIZE_DEFAULT, num_workers=Config.WORKERS_DEFAULT,
             use_gpu=True, gpu_id=Config.GPU_ID_DEFAULT, logr=None,
             data_dir=Config.DATA_DIR_DEFAULT, n_data_samples=Config.NUM_SAMPLES_DEFAULT, cust_graph=False,
             folds=Config.FOLDS_DEFAULT, kid=Config.VALID_K_DEFAULT,
             sample_split=Config.SAMPLE_SPLIT_DEFAULT, stCNN_stride=Config.STCNN_STRIDE_DEFAULT):
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
    logr.log('> Loading DataSet from %s, given %d samples%s\n' %
             (data_dir, n_data_samples, ', using a customized graph' if cust_graph else '')
             )
    dataset = EEGAgeDataSet(data_dir, n_samples=n_data_samples, sample_split=sample_split, cust_graph=cust_graph, folds=folds, valid_k=kid)
    validloader = GraphDataLoader(dataset.valid_set, batch_size=bs, shuffle=False, num_workers=num_workers)
    testloader = GraphDataLoader(dataset.test_set, batch_size=bs, shuffle=False, num_workers=num_workers)
    logr.log('> Validation batches: {}, Test batches: {}\n'.format(len(validloader), len(testloader)))

    # Evaluate
    net.eval()

    # - Validation
    val_metrics = evalMetrics(validloader, device, net)
    logr.log('Validation: %s\n' % metricsMap2Str(val_metrics))
    del validloader
    del val_metrics

    # - Test
    test_metrics = evalMetrics(testloader, device, net)
    logr.log('Test: %s\n' % metricsMap2Str(test_metrics))
    del testloader
    del test_metrics

    # End Evaluation
    logr.log('> Evaluation finished.\n')


if __name__ == '__main__':
    """
        Usage Example:
        python Trainer.py -dr data/xxx -c 4 -m train -net FeedForward
        python Trainer.py -dr data/xxx -c 4 -m eval -e model_save/xx.pth
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-ld', '--log_dir', type=str, default=Config.LOG_DIR_DEFAULT, help='Specify where to create a log file. If log files are not wanted, value will be None'.format(Config.LOG_DIR_DEFAULT))

    parser.add_argument('-lr', '--learning_rate', type=float, default=Config.LEARNING_RATE_DEFAULT, help='Learning rate, default = {}'.format(Config.LEARNING_RATE_DEFAULT))
    parser.add_argument('-me', '--max_epochs', type=int, default=Config.MAX_EPOCHS_DEFAULT, help='Number of epochs to run the trainer, default = {}'.format(Config.MAX_EPOCHS_DEFAULT))
    parser.add_argument('-ef', '--eval_freq', type=int, default=Config.EVAL_FREQ_DEFAULT, help='Frequency of evaluation on the validation set, default = {}'.format(Config.EVAL_FREQ_DEFAULT))
    parser.add_argument('-bs', '--batch_size', type=int, default=Config.BATCH_SIZE_DEFAULT, help='Size of a minibatch, default = {}'.format(Config.BATCH_SIZE_DEFAULT))
    parser.add_argument('-opt', '--optimizer', type=str, default=Config.OPTIMIZER_DEFAULT, help='Optimizer to be used, default = {}'.format(Config.OPTIMIZER_DEFAULT))
    parser.add_argument('-lf', '--loss_function', type=str, default=Config.LOSS_FUNC_DEFAULT, help='Specify which loss function to use, default = {}'.format(Config.LOSS_FUNC_DEFAULT))

    parser.add_argument('-dr', '--data_dir', type=str, default=Config.DATA_DIR_DEFAULT, help='Root directory of the input data, default = {}'.format(Config.DATA_DIR_DEFAULT))
    parser.add_argument('-f', '--folds', type=int, default=Config.FOLDS_DEFAULT, help='Number of folds, default = {}'.format(Config.FOLDS_DEFAULT))
    parser.add_argument('-k', '--k_id', type=int, default=Config.VALID_K_DEFAULT, help='Fold number k (index) used for validation, default = {}'.format(Config.VALID_K_DEFAULT))
    parser.add_argument('-nd', '--num_samples', type=int, default=Config.NUM_SAMPLES_DEFAULT, help='Specify the number of samples to run, default = {}'.format(Config.NUM_SAMPLES_DEFAULT))
    parser.add_argument('-cg', '--customize_graph', type=int, default=Config.CUSTOMIZE_GRAPH_DEFAULT, help='Specify whether to use a customized graph, default = {}'.format(Config.CUSTOMIZE_GRAPH_DEFAULT))

    parser.add_argument('-c', '--cores', type=int, default=Config.WORKERS_DEFAULT, help='number of workers (cores used), default = {}'.format(Config.WORKERS_DEFAULT))
    parser.add_argument('-gpu', '--gpu', type=int, default=Config.USE_GPU_DEFAULT, help='Specify whether to use GPU, default = {}'.format(Config.USE_GPU_DEFAULT))
    parser.add_argument('-gid', '--gpu_id', type=int, default=Config.GPU_ID_DEFAULT, help='Specify which GPU to use, default = {}'.format(Config.GPU_ID_DEFAULT))

    parser.add_argument('-net', '--network', type=str, default=Config.NETWORK_DEFAULT, help='Specify which model/network to use, default = {}, choices = {}'.format(Config.NETWORK_DEFAULT, Config.NETWORKS))
    parser.add_argument('-m', '--mode', type=str, default=Config.MODE_DEFAULT, help='Specify which mode the discriminator runs in (train, eval), default = {}'.format(Config.MODE_DEFAULT))
    parser.add_argument('-e', '--eval', type=str, default=Config.EVAL_DEFAULT, help='Specify the location of saved network to be loaded for evaluation, default = {}'.format(Config.EVAL_DEFAULT))
    parser.add_argument('-md', '--model_save_dir', type=str, default=Config.MODEL_SAVE_DIR_DEFAULT, help='Specify the location of network to be saved, default = {}'.format(Config.MODEL_SAVE_DIR_DEFAULT))

    parser.add_argument('-fd', '--feature_dim', type=int, default=Config.FEAT_DIM_DEFAULT, help='Specify the feature dimension, default = {}'.format(Config.FEAT_DIM_DEFAULT))
    parser.add_argument('-hd', '--hidden_dim', type=int, default=Config.HIDDEN_DIM_DEFAULT, help='Specify the hidden dimension, default = {}'.format(Config.HIDDEN_DIM_DEFAULT))

    # For testing
    parser.add_argument('-s', '--sample_split', type=int, default=Config.SAMPLE_SPLIT_DEFAULT, help='Specify sample split in preprocessing (for testing), default = {}'.format(Config.SAMPLE_SPLIT_DEFAULT))
    parser.add_argument('-sts', '--stcnn_stride', type=int, default=Config.STCNN_STRIDE_DEFAULT, help='Specify the stride for StCNN (for testing), default = {}'.format(Config.STCNN_STRIDE_DEFAULT))

    FLAGS, unparsed = parser.parse_known_args()

    # Starts a log file in the specified directory
    logger = Logger(activate=True, logging_folder=FLAGS.log_dir) if FLAGS.log_dir else Logger(activate=False)

    # Samples: if not specified, calculate
    num_samples = int(Config.NUM_SUBJECTS * FLAGS.sample_split) if FLAGS.num_samples == -1 else FLAGS.num_samples

    working_mode = FLAGS.mode
    if working_mode == 'train':
        train(lr=FLAGS.learning_rate, bs=FLAGS.batch_size, ep=FLAGS.max_epochs,
              eval_freq=FLAGS.eval_freq, opt=FLAGS.optimizer, num_workers=FLAGS.cores,
              use_gpu=(FLAGS.gpu == 1), gpu_id=FLAGS.gpu_id, logr=logger,
              data_dir=FLAGS.data_dir, n_data_samples=num_samples, cust_graph=(FLAGS.customize_graph == 1),
              model=FLAGS.network, model_save_dir=FLAGS.model_save_dir,
              loss_function=FLAGS.loss_function,
              feat_dim=FLAGS.feature_dim, hidden_dim=FLAGS.hidden_dim,
              folds=FLAGS.folds, kid=FLAGS.k_id,
              sample_split=FLAGS.sample_split, stCNN_stride=FLAGS.stcnn_stride)
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
                 use_gpu=(FLAGS.gpu == 1), gpu_id=FLAGS.gpu_id, logr=logger,
                 data_dir=FLAGS.data_dir, n_data_samples=num_samples, cust_graph=(FLAGS.customize_graph == 1),
                 folds=FLAGS.folds, kid=FLAGS.k_id,
                 sample_split=FLAGS.sample_split, stCNN_stride=FLAGS.stcnn_stride)
        logger.close()
    else:
        sys.stderr.write('Please specify the working mode (train/eval)\n')
        logger.close()
        exit(-2)
