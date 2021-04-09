import pickle as pickle
import os
import math
import unicodedata

import numpy as np
from scipy.signal import argrelextrema

import os
import time
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
#from dataset import DanceDataset, paired_collate_fn
from model import Encoder, Decoder, Model
from utils.log import Logger
from ddc_data_prep import prepare_train_batch
#from utils.functional import str2bool, load_data
import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=10000)
parser.add_argument('--output_dir', metavar='PATH',
                        default='checkpoints/layers2_win100_schedule100_condition10_detach')
parser.add_argument('--tensorboard', action='store_false')
parser.add_argument('--log_per_updates', type=int, metavar='N', default=1,
                        help='log model loss per x updates (mini-batches).')
args = parser.parse_args()

global log
log = Logger(args)
#transformer helper function

def train(model, training_data, optimizer, device, args, log):
    """ Start training """
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    updates = 0  # global step

    for epoch_i in range(1, args.epoch + 1):
        log.set_progress(epoch_i, len(training_data))
        model.train()

        for batch_i, batch in enumerate(training_data):
            # prepare data this is all wrong for the DDC data style
            # todo 
            
            hidden, out_frame, out_seq = model.module.init_decoder_hidden(tgt_seq.size(0))

            # forward
            optimizer.zero_grad()

            output = model(src_seq, src_pos, tgt_seq, hidden, out_frame, out_seq, epoch_i)

            # backward
            loss = criterion(output, gold_seq)
            loss.backward()

            # update parameters
            optimizer.step()

            stats = {
                'updates': updates,
                'loss': loss.item()
            }
            log.update(stats)
            updates += 1

        checkpoint = {
            'model': model.state_dict(),
            'args': args,
            'epoch': epoch_i
        }

        if epoch_i % args.save_per_epochs == 0 or epoch_i == 1:
            filename = os.path.join(args.output_dir, f'epoch_{epoch_i}.pt')
            torch.save(checkpoint, filename)
#ddc helper functions needed. 
#purpose, load the DDC style features and data into a simple pytorch framework this involvces ddc style audio feature extraction
def open_dataset_fps(*args):
    datasets = []
    for data_fp in args:
        if not data_fp:
            datasets.append([])
            continue

        with open(data_fp, 'r') as f:
            song_fps = f.read().split()
        dataset = []
        for song_fp in song_fps:
            with open(song_fp, 'rb') as f:
                dataset.append(pickle.load(f))
        datasets.append(dataset)
    return datasets[0] if len(datasets) == 1 else datasets

def stride_csv_arg_list(arg, stride, cast=int):
    assert stride > 0
    l = [x for x in [x.strip() for x in arg.split(',')] if bool(x)]
    l = [cast(x) for x in l]
    assert len(l) % stride == 0
    result = []
    for i in range(0, len(l), stride):
        if stride == 1:
            subl = l[i]
        else:
            subl = tuple(l[i:i + stride])
        result.append(subl)
    return result
def select_channels(dataset, channels):
    for i, (song_metadata, song_features, song_charts) in enumerate(dataset):
        song_features_selected = song_features[:, :, channels]
        dataset[i] = (song_metadata, song_features_selected, song_charts)
        for chart in song_charts:
            chart.song_features = song_features_selected

def flatten_dataset_to_charts(dataset):
    return [item for sublist in [x[2] for x in dataset] for item in sublist]
#manually loading ddc data    
train_txt_fp= "/home/nerd/ddt/dance-dance-transformation/data/chart_pkl/mel80hop441/fraxil_train.txt"#('train_txt_fp', '', 'Training dataset txt file with a list of pickled song files')
valid_txt_fp = "/home/nerd/ddt/dance-dance-transformation/data/chart_pkl/mel80hop441/fraxil_valid.txt" #tf.app.flags.DEFINE_string('valid_txt_fp', '', 'Eval dataset txt file with a list of pickled song files')
test_txt_fp = "/home/nerd/ddt/dance-dance-transformation/data/chart_pkl/mel80hop441/fraxil_test.txt" #tf.app.flags.DEFINE_string('test_txt_fp', '', 'Test dataset txt file with a list of pickled song files')
print('Loading data')
train_data, valid_data, test_data = open_dataset_fps(train_txt_fp, valid_txt_fp, test_txt_fp)

    # Select channels
audio_select_channels = "0,1,2"
if audio_select_channels:
        channels = stride_csv_arg_list(audio_select_channels, 1, int)
        print('Selecting channels {} from data'.format(channels))
        for data in [train_data, valid_data, test_data]:
            select_channels(data, channels)
charts_train = flatten_dataset_to_charts(train_data)
charts_valid = flatten_dataset_to_charts(valid_data)
charts_test = flatten_dataset_to_charts(test_data)
# we now have a list object of the ddc "chart" object, and we need to generate data

# 
feats_audio, feats_other, targets, target_weights = prepare_train_batch(charts_train)
print(len(charts_train))
print(charts_train[0].song_metadata['title'])
print(charts_train[0])
#for chart in range(len(charts_train)):
#    print(charts_train[chart].song_metadata['title'])
print(len(feats_audio)) #24 the batch size
print(feats_audio.shape) #(24, 1, 3, 80, 3)
print(feats_other.shape) # (24, 1, 0)
print(targets.shape) # (24, 1)

print(targets)
#print(feats_audio[0]) # 
#print(train_data[0])
#print(type(train_data[0]))
#setting transformer variables. 
cuda= "yes"

max_seq_len= 4500
d_frame_vec = 50
frame_emb_size =10
n_layers = 2
n_head =2
d_k = 64
d_v = 64
d_model = frame_emb_size
d_inner= 1024
dropout= .1
d_pose_vec = 50
pose_emb_size= 50
condition_step =10
sliding_windown_size =100
lambda_v = 0.01
lr = .01
device = torch.device('cuda' if cuda =="yes" else 'cpu')

#setup encoder
encoder = Encoder(max_seq_len=max_seq_len,
                      input_size=d_frame_vec,
                      d_word_vec=frame_emb_size,
                      n_layers=n_layers,
                      n_head=n_head,
                      d_k=d_k,
                      d_v=d_v,
                      d_model=d_model,
                      d_inner=d_inner,
                      dropout=dropout)
#setup decoder
decoder = Decoder(input_size=d_pose_vec,
                      d_word_vec=pose_emb_size,
                      hidden_size=d_inner,
                      encoder_d_model=d_model,
                      dropout=dropout)
#setup model
model = Model(encoder, decoder,
                  condition_step=condition_step,
                  sliding_windown_size=sliding_windown_size,
                  lambda_v=lambda_v,
                  device=device)

#print(model)
model = nn.DataParallel(model).to(device)
optimizer = optim.Adam(filter(
        lambda x: x.requires_grad, model.module.parameters()), lr=lr)
train(model, train_data, optimizer, device, args, log)
