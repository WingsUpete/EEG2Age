import os


if __name__ == '__main':
    # K-Fold
    inputs = {
        'network': 'FeedForward',
        'folds': 5
    }
    for kid in range(inputs['folds']):
        os.system('python Trainer.py -dr data/ -m train -net %s -f %d -k %d'
                  % (inputs['network'], inputs['folds'], kid))
