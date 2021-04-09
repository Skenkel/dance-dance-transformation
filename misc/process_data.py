import numpy as np
# block from https://github.com/stonyhu/DanceRevolution/blob/master/prepro.py to pull audio code
import os
import sys
import json
import random
import argparse
import essentia
import essentia.streaming
from essentia.standard import *
import librosa
import numpy as np
from extractor import FeatureExtractor

parser = argparse.ArgumentParser()
parser.add_argument('--input_audio_dir', type=str, default='data/audio')
parser.add_argument('--input_chart_dir', type=str, default='data/json')

parser.add_argument('--train_dir', type=str, default='data/train')
parser.add_argument('--test_dir', type=str, default='data/test')

parser.add_argument('--sampling_rate', type=int, default=15360)
args = parser.parse_args()

extractor = FeatureExtractor()

if not os.path.exists(args.train_dir):
    os.mkdir(args.train_dir)
if not os.path.exists(args.test_dir):
    os.mkdir(args.test_dir)



def extract_acoustic_feature(input_audio_dir):
    print('---------- Extract features from raw audio ----------')
    musics = []
    # onset_beats = []
    audio_fnames = sorted(os.listdir(input_audio_dir))
    # audio_fnames = audio_fnames[:20]  # for debug
    print(f'audio_fnames: {audio_fnames}')

    for audio_fname in audio_fnames:
        audio_file = os.path.join(input_audio_dir, audio_fname)
        print(f'Process -> {audio_file}')
        ### load audio ###
        sr = args.sampling_rate
        # sr = 48000
        loader = essentia.standard.MonoLoader(filename=audio_file, sampleRate=sr)
        audio = loader()
        audio = np.array(audio).T

        melspe_db = extractor.get_melspectrogram(audio, sr)
        mfcc = extractor.get_mfcc(melspe_db)
        mfcc_delta = extractor.get_mfcc_delta(mfcc)
        # mfcc_delta2 = get_mfcc_delta2(mfcc)

        audio_harmonic, audio_percussive = extractor.get_hpss(audio)
        # harmonic_melspe_db = get_harmonic_melspe_db(audio_harmonic, sr)
        # percussive_melspe_db = get_percussive_melspe_db(audio_percussive, sr)
        chroma_cqt = extractor.get_chroma_cqt(audio_harmonic, sr)
        # chroma_stft = extractor.get_chroma_stft(audio_harmonic, sr)

        onset_env = extractor.get_onset_strength(audio_percussive, sr)
        tempogram = extractor.get_tempogram(onset_env, sr)
        onset_beat = extractor.get_onset_beat(onset_env, sr)
        # onset_tempo, onset_beat = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        # onset_beats.append(onset_beat)

        onset_env = onset_env.reshape(1, -1)

        feature = np.concatenate([
            # melspe_db,
            mfcc,
            mfcc_delta,
            # mfcc_delta2,
            # harmonic_melspe_db,
            # percussive_melspe_db,
            # chroma_stft,
            chroma_cqt,
            onset_env,
            onset_beat,
            tempogram
        ], axis=0)

        feature = feature.transpose(1, 0)
        print(f'acoustic feature -> {feature.shape}')
        musics.append(feature.tolist())
    

    return musics
 # block from https://github.com/chrisdonahue/ddc/blob/master/dataset/extract_json.py
 import glob
import logging as smlog
import os
import traceback

from smdataset.abstime import calc_note_beats_and_abs_times
from smdataset.parse import parse_sm_txt

_ATTR_REQUIRED = ['offset', 'bpms', 'notes']

if __name__ == '__main__':
    import argparse
    from collections import OrderedDict
    import json
    json.encoder.FLOAT_REPR = lambda f: ('%.6f' % f)
    from util import ez_name, get_subdirs

    parser = argparse.ArgumentParser()
    parser.add_argument('packs_dir', type=str, help='Directory of packs (organized like Stepmania songs folder)')
    parser.add_argument('json_dir', type=str, help='Output JSON directory')
    parser.add_argument('--itg', dest='itg', action='store_true', help='If set, subtract 9ms from offset')
    parser.add_argument('--choose', dest='choose', action='store_true', help='If set, choose from list of packs')

    parser.set_defaults(
        itg=False,
        choose=False)

    args = parser.parse_args()

    pack_names = get_subdirs(args.packs_dir, args.choose)
    pack_dirs = [os.path.join(args.packs_dir, pack_name) for pack_name in pack_names]
    pack_sm_globs = [os.path.join(pack_dir, '*', '*.sm') for pack_dir in pack_dirs]

    if not os.path.isdir(args.json_dir):
        os.mkdir(args.json_dir)

    pack_eznames = set()
    for pack_name, pack_sm_glob in zip(pack_names, pack_sm_globs):
        pack_sm_fps = sorted(glob.glob(pack_sm_glob))
        pack_ezname = ez_name(pack_name)
        if pack_ezname in pack_eznames:
            raise ValueError('Pack name conflict: {}'.format(pack_ezname))
        pack_eznames.add(pack_ezname)

        if len(pack_sm_fps) > 0:
            pack_outdir = os.path.join(args.json_dir, pack_ezname)
            if not os.path.isdir(pack_outdir):
                os.mkdir(pack_outdir)

        sm_eznames = set()
        for sm_fp in pack_sm_fps:
            sm_name = os.path.split(os.path.split(sm_fp)[0])[1]
            sm_ezname = ez_name(sm_name)
            if sm_ezname in sm_eznames:
                raise ValueError('Song name conflict: {}'.format(sm_ezname))
            sm_eznames.add(sm_ezname)

            with open(sm_fp, 'r') as sm_f:
                sm_txt = sm_f.read()

            # parse file
            try:
                sm_attrs = parse_sm_txt(sm_txt)
            except ValueError as e:
                smlog.error('{} in\n{}'.format(e, sm_fp))
                continue
            except Exception as e:
                smlog.critical('Unhandled parse exception {}'.format(traceback.format_exc()))
                raise e

            # check required attrs
            try:
                for attr_name in _ATTR_REQUIRED:
                    if attr_name not in sm_attrs:
                        raise ValueError('Missing required attribute {}'.format(attr_name))
            except ValueError as e:
                smlog.error('{}'.format(e))
                continue

            # handle missing music
            root = os.path.abspath(os.path.join(sm_fp, '..'))
            music_fp = os.path.join(root, sm_attrs.get('music', ''))
            if 'music' not in sm_attrs or not os.path.exists(music_fp):
                music_names = []
                sm_prefix = os.path.splitext(sm_name)[0]

                # check directory files for reasonable substitutes
                for filename in os.listdir(root):
                    prefix, ext = os.path.splitext(filename)
                    if ext.lower()[1:] in ['mp3', 'ogg']:
                        music_names.append(filename)

                try:
                    # handle errors
                    if len(music_names) == 0:
                        raise ValueError('No music files found')
                    elif len(music_names) == 1:
                        sm_attrs['music'] = music_names[0]
                    else:
                        raise ValueError('Multiple music files {} found'.format(music_names))
                except ValueError as e:
                    smlog.error('{}'.format(e))
                    continue

                music_fp = os.path.join(root, sm_attrs['music'])

            bpms = sm_attrs['bpms']
            offset = sm_attrs['offset']
            if args.itg:
                # Many charters add 9ms of delay to their stepfiles to account for ITG r21/r23 global delay
                # see http://r21freak.com/phpbb3/viewtopic.php?f=38&t=12750
                offset -= 0.009
            stops = sm_attrs.get('stops', [])

            out_json_fp = os.path.join(pack_outdir, '{}_{}.json'.format(pack_ezname, sm_ezname))
            out_json = OrderedDict([
                ('sm_fp', os.path.abspath(sm_fp)),
                ('music_fp', os.path.abspath(music_fp)),
                ('pack', pack_name),
                ('title', sm_attrs.get('title')),
                ('artist', sm_attrs.get('artist')),
                ('offset', offset),
                ('bpms', bpms),
                ('stops', stops),
                ('charts', [])
            ])

            for idx, sm_notes in enumerate(sm_attrs['notes']):
                note_beats_and_abs_times = calc_note_beats_and_abs_times(offset, bpms, stops, sm_notes[5])
                notes = {
                    'type': sm_notes[0],
                    'desc_or_author': sm_notes[1],
                    'difficulty_coarse': sm_notes[2],
                    'difficulty_fine': sm_notes[3],
                    'notes': note_beats_and_abs_times,
                }
                out_json['charts'].append(notes)

            with open(out_json_fp, 'w') as out_f:
                try:
                    out_f.write(json.dumps(out_json))
                except UnicodeDecodeError:
                    smlog.error('Unicode error in {}'.format(sm_fp))
                    continue

            print 'Parsed {} - {}: {} charts'.format(pack_name, sm_name, len(out_json['charts']))
