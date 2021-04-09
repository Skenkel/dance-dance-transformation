import random
import torch
import numpy as np

#todo, need to proces out the chart diffulty as an int
# to do, need to process out certain charts 
def prepare_train_batch(charts, batch_size= 24, **kwargs):
    # process kwargs
    exclude_kwarg_names = ['exclude_onset_neighbors', 'exclude_pre_onsets', 'exclude_post_onsets', 'include_onsets']
    exclude_kwargs = {k:v for k,v in list(kwargs.items()) if k in exclude_kwarg_names}
    feat_kwargs = {k:v for k,v in list(kwargs.items()) if k not in exclude_kwarg_names}
    rnn_nunroll = 1
    #np_dtype = torch.float32
    np_dtype= np.float32
    batch_feats_audio = []
    batch_feats_other = []
    batch_targets = []
    batch_target_weights = []
    for _ in range(batch_size):
            chart = charts[random.randint(0, len(charts) - 1)]
            frame_idx = chart.sample(1, **exclude_kwargs)[0]

            subseq_start = frame_idx - (rnn_nunroll - 1)
            audio, other, target = chart.get_subsequence(subseq_start, rnn_nunroll, np_dtype, **feat_kwargs)

            batch_feats_audio.append(audio)
            batch_feats_other.append(other)
            batch_targets.append(target)
            weight = target[:] #using the pos target weight strategy
            batch_target_weights.append(weight)

    batch_feats_audio = np.array(batch_feats_audio, dtype=np_dtype)
    batch_feats_other = np.array(batch_feats_other, dtype=np_dtype)
    batch_targets = np.array(batch_targets, dtype=np_dtype)
    batch_target_weights = np.array(batch_target_weights, dtype=np_dtype)

    return batch_feats_audio, batch_feats_other, batch_targets, batch_target_weights