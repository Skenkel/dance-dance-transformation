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

from model import Encoder, Decoder, Model
from utils.log import Logger
from ddc_data_prep import prepare_train_batch

import warnings
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--log_per_updates', type=int, metavar='N', default=1,
                        help='log model loss per x updates (mini-batches).')
parser.add_argument('--output_dir', metavar='PATH',
                        default='checkpoints/layers2_win100_schedule100_condition10_detach')
parser.add_argument('--tensorboard', action='store_false')

args = parser.parse_args()

global log
log = Logger(args)


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
# you will need to update the file location and contents if your directory isn't setup the way mine is. 
#     
train_txt_fp= "/home/nerd/ddt/dance-dance-transformation/data/chart_pkl/mel80hop441/fraxil_train.txt"#('train_txt_fp', '', 'Training dataset txt file with a list of pickled song files')
valid_txt_fp = "/home/nerd/ddt/dance-dance-transformation/data/chart_pkl/mel80hop441/fraxil_valid.txt" #tf.app.flags.DEFINE_string('valid_txt_fp', '', 'Eval dataset txt file with a list of pickled song files')
test_txt_fp = "/home/nerd/ddt/dance-dance-transformation/data/chart_pkl/mel80hop441/fraxil_test.txt" #tf.app.flags.DEFINE_string('test_txt_fp', '', 'Test dataset txt file with a list of pickled song files')
print('Loading data')
train_data, valid_data, test_data = open_dataset_fps(train_txt_fp, valid_txt_fp, test_txt_fp)

    # Select channels
#audio_select_channels = "0,1"
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
# we do this using the chart class from DDC. 

for chart_num in range(len(charts_train)):
    chart = charts_train[chart_num]
    print(chart.song_metadata['title']) 
    print(chart.nframes)
    print(chart.first_onset) #frame of the first step. 
    print(chart.metadata[1]) #this is the difficulty number to embed later. 
    chart_audio, chart_other, chart_target = chart.get_subsequence(1,chart.nframes, np.float32) # 
    print(chart_audio.shape) #frames by 3 by 80 by numver of audio channels 
    print(chart_target.shape) #frames
    #print(chart_target) #array of targets
    #print(chart_target[738]) # should always be 1 because it's the first step of the chart. 
    print(chart_target.sum())
    #print(charts_train[chart].nframes)
    #print(charts_train[chart].onsets)
    #print(charts_train[chart].first_onset)


#print(targets)
#chart = charts_train[0]
#print(chart.song_metadata['title']) 
#print(chart.nframes)
#print(chart.first_onset) #frame of the first step. 
#print(chart.metadata[1]) #this is the difficulty number to embed later. 
#chart_audio, chart_other, chart_target = chart.get_subsequence(1,chart.nframes, np.float32) # 
#print(chart_audio.shape) #frames by 3 by 80 by 3 
#print(chart_target.shape)
#print(chart_target) #array of targets
#print(chart_target[738]) # should always be 1 because it's the first step of the chart. 
#print(chart_target.sum())