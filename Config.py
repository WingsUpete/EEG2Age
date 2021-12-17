# DEBUG FLAGS
TRAIN_JUST_ONE_BATCH = False
TRAIN_JUST_ONE_ROUND = False
PROFILE = False
CHECK_GRADS = False

# Basic
LEARNING_RATE_DEFAULT = 1e-2    # 0.01
MAX_EPOCHS_DEFAULT = 200
EVAL_FREQ_DEFAULT = 5
BATCH_SIZE_DEFAULT = 5
WORKERS_DEFAULT = 4

OPTIMIZER_DEFAULT = 'ADAM'
WEIGHT_DECAY_DEFAULT = 0.01

DATA_DIR_DEFAULT = 'data/EEG_age_data/'
LOG_DIR_DEFAULT = 'log/'

USE_GPU_DEFAULT = 1
GPU_ID_DEFAULT = 0

NETWORK_DEFAULT = 'BAPM'
NETWORKS = ['FeedForward', 'BAPM', 'GRUNet']          # TODO

MODE_DEFAULT = 'train'
EVAL_DEFAULT = 'model_save/eval.pt'   # should be a model file name
MODEL_SAVE_DIR_DEFAULT = 'model_save/'

MAX_NORM_DEFAULT = 10.0

NUM_HEADS_DEFAULT = 3
FEAT_DIM_DEFAULT = 1
HIDDEN_DIM_DEFAULT = 1

LOSS_FUNC_DEFAULT = 'SmoothL1Loss'
SMOOTH_L1_LOSS_BETA_DEFAULT = 10

FOLDS_DEFAULT = 5
VALID_K_DEFAULT = -1

# DIY
NUM_NODES = 63
NUM_SAMPLES = 111
NUM_TIMESTAMPS = 61440
