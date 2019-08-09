from flask import Flask, request
import json

import logging
import os
import random
import string
import sys

import sentencepiece as spm
import torch
import torchaudio
import numpy as np
from fairseq import options, progress_bar, utils, tasks
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.utils import import_user_module

app = Flask(__name__)

import argparse

# manually preparing argument
"""
Complete list of arguments for fairseq is as follows:

```
Namespace(beam=40, bpe=None, cpu=False, criterion='cross_entropy', ctc=False, data='./data', dataset_impl=None, diverse_bea
m_groups=-1, diverse_beam_strength=0.5, force_anneal=None, fp16=False, fp16_init_scale=128, fp16_scale_tolerance=0.0, fp16_
scale_window=None, gen_subset='test', kspmodel=None, lenpen=1, lm_weight=0.2, log_format=None, log_interval=1000, lr_schedu
ler='fixed', lr_shrink=0.1, match_source_len=False, max_len_a=0, max_len_b=200, max_sentences=None, max_tokens=10000000, me
mory_efficient_fp16=False, min_len=1, min_loss_scale=0.0001, model_overrides='{}', momentum=0.99, nbest=1, no_beamable_mm=F
alse, no_early_stop=False, no_progress_bar=False, no_repeat_ngram_size=0, num_shards=1, num_workers=1, optimizer='nag', pat
h='./data/checkpoint_avg_60_80.pt', prefix_size=0, print_alignment=False, quiet=False, remove_bpe=None, replace_unk=None, r
equired_batch_size_multiple=8, results_path=None, rnnt=False, rnnt_decoding_type='greedy', rnnt_len_penalty=-0.5, sacrebleu
=False, sampling=False, sampling_topk=-1, sampling_topp=-1.0, score_reference=False, seed=1, shard_id=0, skip_invalid_size_
inputs_valid_test=False, task='speech_recognition', tbmf_wrapper=False, temperature=1.0, tensorboard_logdir='', threshold_l
oss_scale=None, tokenizer=None, unkpen=0, unnormalized=False, user_dir='../fairseq/examples/speech_recognition', warmup_upd
ates=0, weight_decay=0.0, wfstlm=None)
```

"""
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--max-tokens', type=int, default=10000000)
parser.add_argument('--nbest', type=int, default=1)
parser.add_argument('--path', default='./data/checkpoint_avg_60_80.pt')
parser.add_argument('--beam', type=int, default=40)
parser.add_argument('--user_dir', default='../fairseq/examples/speech_recognition')
parser.add_argument('--task', default='speech_recognition')
parser.add_argument('--data', default='./data')
parser.add_argument('--model_overrides', default='')
parser.add_argument('--no_beamable_mm', default=False)
parser.add_argument('--print_alignment', default=False)

args = parser.parse_args()
print("ARGS:", args)

# 1. Load the model

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def optimize_models(args, models):
    """Optimize ensemble for generation
    """
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )

    model.to(dev)

#def model_load(path, no_beamable_mm, beam, print_alignment, model_overrides):
def model_load(args):
    # Load ensemble
    logger.info("| loading model(s) from {}".format(args.path))
    models, _model_args = utils.load_ensemble_for_inference(
        args.path.split(":"),
        task,
        model_arg_overrides={}
    )
    optimize_models(args, models)

    # Initialize generator
    generator = task.build_generator(args)

    sp = spm.SentencePieceProcessor()
    sp.Load(os.path.join(args.data, 'spm.model'))
    
    return models, sp, generator

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load dataset splits
task = tasks.setup_task(args)
    
models, sp, generator = model_load(args)

def process_predictions(args, hypos, sp, tgt_dict):
    res = []
    for hypo in hypos[: min(len(hypos), args.nbest)]:
        hyp_pieces = tgt_dict.string(hypo["tokens"].int().cpu())
        hyp_words = sp.DecodePieces(hyp_pieces.split())
        res.append(hyp_words)
    return res

def calc_mean_invstddev(feature):
    if len(feature.shape) != 2:
        raise ValueError("We expect the input feature to be 2-D tensor")
    mean = np.mean(feature, axis=0)
    var = np.var(feature, axis=0)
    # avoid division by ~zero
    if var.any() < sys.float_info.epsilon:
        return mean, 1.0 / (np.sqrt(var) + sys.float_info.epsilon)
    return mean, 1.0 / np.sqrt(var)


def calcMN(features):
    mean, invstddev = calc_mean_invstddev(features)
    res = (features - mean) * invstddev
    return res

def transcribe(waveform, args, task, generator, models, sp, tgt_dict):
    r"""
    CUDA_VISIBLE_DEVICES=0 python infer_asr.py /Users/jamarshon/Documents/downloads/ \
        --task speech_recognition --max-tokens 10000000 --nbest 1 --path \
        /Users/jamarshon/Downloads/checkpoint_avg_60_80.pt --beam 20
    """
    num_features = 80
    output = torchaudio.compliance.kaldi.fbank(waveform, num_mel_bins=num_features)
    output_cmvn = calcMN(output.cpu().detach().numpy())

    # size (m, n)
    source = torch.tensor(output_cmvn)
    source = source.to(dev)
    frames_lengths = torch.LongTensor([source.size(0)])

    # size (1, m, n). In general, if source is (x, m, n), then hypos is (x, ...)
    source.unsqueeze_(0)
    sample = {'net_input': {'src_tokens': source, 'src_lengths': frames_lengths}}

    hypos = task.inference_step(generator, models, sample)

    assert len(hypos) == 1
    transcription = []
    print(hypos)
    for i in range(len(hypos)):
        # Process top predictions
        hyp_words = process_predictions(args, hypos[i], sp, tgt_dict)
        transcription.append(hyp_words)

    print('transcription:', transcription)
    return transcription

    
# 2. Write a function for inference - wav file
def infer(wav_file):
    waveform, sample_rate = torchaudio.load_wav(wav_file)
    waveform = waveform.mean(0, True)
    waveform = torchaudio.transforms.Resample(orig_freq=sample_rate,new_freq=16000)(waveform)
    print("waveform", waveform.shape)
    import time
    print(sample_rate, waveform.shape)
    start = time.time()
    tgt_dict = task.target_dictionary
    transcription = transcribe(waveform, args, task, generator, models, sp, tgt_dict)
    end = time.time()
    print(end - start)
    return transcription
    
    
# 3. Define a route for inference
@app.route('/transcribe', methods=['POST'])
def transcribe_route():
    # Get the file out from the request
    print(request.files)
    print('wav_file', request.files['audio'])
    wav_file = request.files['audio']
    wav_file_name = './tmp/' + wav_file.filename
    wav_file.save(wav_file_name)
    print("wav_file_name", wav_file_name)
    transcription = infer(wav_file_name)
    print("translated transcript>>", transcription)

    # Do the inference, get the result

    # Return json
    return json.dumps({'success': True, 'transcription': transcription[0][0]})



@app.route('/')
def hello_world():
        return 'Hello, World!'

if __name__ == '__main__':
    #model_path = ''
    #model_load(model_path)

    app.run(host='0.0.0.0', port=8090)
    


