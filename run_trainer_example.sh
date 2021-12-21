#!/bin/sh
ulimit -n 65535

### Preprocess ###
cd preprocess/
python GraphBuilder.py -d ../data/EEG_age_data_raw/ -c 4 -s 16 -o ../data/EEG_age_data_s16/
python GraphCustomizer.py -g ../data/D_lobes.txt -o ../data/
cd ../



### BAPM ###
python Trainer.py -dr data/EEG_age_data_s16/ -s 16 -c 4 -m train -net BAPM
#python Trainer.py -dr data/EEG_age_data_s16/ -s 16 -c 4 -m eval -e model_save/20211219_23_35_13.pth



##### Baselines #####
### FeedForward ###
#python Trainer.py -dr data/EEG_age_data_s16/ -s 16 -c 4 -m train -net FeedForward
#python Trainer.py -dr data/EEG_age_data_s16/ -s 16 -c 4 -m eval -e model_save/xxx.pth

### GRUNet ###
#python Trainer.py -dr data/EEG_age_data_s16/ -s 16 -c 4 -m train -net GRUNet
#python Trainer.py -dr data/EEG_age_data_s16/ -s 16 -c 4 -m eval -e model_save/xxx.pth