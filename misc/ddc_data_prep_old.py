# the ddc code which pulls from 
def prepare_train_batch(self, charts, randomize_charts=False, **kwargs):
    # process kwargs
    exclude_kwarg_names = ['exclude_onset_neighbors', 'exclude_pre_onsets', 'exclude_post_onsets', 'include_onsets']
    exclude_kwargs = {k:v for k,v in list(kwargs.items()) if k in exclude_kwarg_names}
    feat_kwargs = {k:v for k,v in list(kwargs.items()) if k not in exclude_kwarg_names}

    # pick random chart and sample balanced classes
    if randomize_charts:
        del exclude_kwargs['exclude_pre_onsets']
        del exclude_kwargs['exclude_post_onsets']
        del exclude_kwargs['include_onsets']
        if self.do_rnn:
            exclude_kwargs['nunroll'] = self.rnn_nunroll

        # create batch
        batch_feats_audio = []
        batch_feats_other = []
        batch_targets = []
        batch_target_weights = []
        for _ in range(self.batch_size):
            chart = charts[random.randint(0, len(charts) - 1)]
            frame_idx = chart.sample(1, **exclude_kwargs)[0]

            subseq_start = frame_idx - (self.rnn_nunroll - 1)

            if self.target_weight_strategy == 'pos' or self.target_weight_strategy == 'posbal':
                target_sum = 0.0
                while target_sum == 0.0:
                    audio, other, target = chart.get_subsequence(subseq_start, self.rnn_nunroll, np_dtype, **feat_kwargs)
                    target_sum = np.sum(target)
                    if target_sum == 0.0:
                        frame_idx = chart.sample_blanks(1, **exclude_kwargs).pop()
                        subseq_start = frame_idx - (self.rnn_nunroll - 1)
            else:
                feat_kwargs['zack_hack_div_2'] = self.zack_hack_div_2
                audio, other, target = chart.get_subsequence(subseq_start, self.rnn_nunroll, np_dtype, **feat_kwargs)

            batch_feats_audio.append(audio)
            batch_feats_other.append(other)
            batch_targets.append(target)

            if self.target_weight_strategy == 'rect':
                weight = np.ones_like(target)
            elif self.target_weight_strategy == 'last':
                weight = np.zeros_like(target)
                weight[-1] = 1.0
            elif self.target_weight_strategy == 'pos':
                weight = target[:]
            elif self.target_weight_strategy == 'posbal':
                negs = set(np.where(target == 0)[0])
                negs_weighted = random.sample(negs, int(np.sum(target)))
                weight = target[:]
                weight[list(negs_weighted)] = 1.0
            batch_target_weights.append(weight)

        # create return arrays
        batch_feats_audio = np.array(batch_feats_audio, dtype=np_dtype)
        batch_feats_other = np.array(batch_feats_other, dtype=np_dtype)
        batch_targets = np.array(batch_targets, dtype=np_dtype)
        batch_target_weights = np.array(batch_target_weights, dtype=np_dtype)

        return batch_feats_audio, batch_feats_other, batch_targets, batch_target_weights
    else:
        chart = charts[random.randint(0, len(charts) - 1)]
        chart_nonsets = chart.get_nonsets()
        if exclude_kwargs.get('include_onsets', False):
            npos = 0
            nneg = self.batch_size
        else:
            npos = min(self.batch_size // 2, chart_nonsets)
            nneg = self.batch_size - npos
        samples = chart.sample_onsets(npos) + chart.sample_blanks(nneg, **exclude_kwargs)
        random.shuffle(samples)

        # create batch
        batch_feats_audio = []
        batch_feats_other = []
        batch_targets = []
        batch_target_weights = []
        for frame_idx in samples:
            subseq_start = frame_idx - (self.rnn_nunroll - 1)

            if self.target_weight_strategy == 'pos' or self.target_weight_strategy == 'posbal':
                target_sum = 0.0
                while target_sum == 0.0:
                    audio, other, target = chart.get_subsequence(subseq_start, self.rnn_nunroll, np_dtype, **feat_kwargs)
                    target_sum = np.sum(target)
                    if target_sum == 0.0:
                        frame_idx = chart.sample_blanks(1, **exclude_kwargs).pop()
                        subseq_start = frame_idx - (self.rnn_nunroll - 1)
            else:
                feat_kwargs['zack_hack_div_2'] = self.zack_hack_div_2
                audio, other, target = chart.get_subsequence(subseq_start, self.rnn_nunroll, np_dtype, **feat_kwargs)

            batch_feats_audio.append(audio)
            batch_feats_other.append(other)
            batch_targets.append(target)

            if self.target_weight_strategy == 'rect':
                weight = np.ones_like(target)
            elif self.target_weight_strategy == 'last':
                weight = np.zeros_like(target)
                weight[-1] = 1.0
            elif self.target_weight_strategy == 'pos':
                weight = target[:]
            elif self.target_weight_strategy == 'posbal':
                negs = set(np.where(target == 0)[0])
                negs_weighted = random.sample(negs, int(np.sum(target)))
                weight = target[:]
                weight[list(negs_weighted)] = 1.0
            batch_target_weights.append(weight)

        # create return arrays
        batch_feats_audio = np.array(batch_feats_audio, dtype=np_dtype)
        batch_feats_other = np.array(batch_feats_other, dtype=np_dtype)
        batch_targets = np.array(batch_targets, dtype=np_dtype)
        batch_target_weights = np.array(batch_target_weights, dtype=np_dtype)

        return batch_feats_audio, batch_feats_other, batch_targets, batch_target_weights