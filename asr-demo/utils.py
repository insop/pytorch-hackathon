from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import librosa
import math
from PIL import Image

import librosa.display

class conf:
    sampling_rate = 44100
    duration = 4 # sec
    hop_length = 347*duration # to make time steps 128
    fmin = 20
    fmax = sampling_rate // 2
    n_mels = 128
    n_fft = n_mels * 20
    padmode = 'constant'
    samples = sampling_rate * duration

def read_audio(conf, pathname, trim_long_data=True):
    y, sr = librosa.load(pathname, sr=conf.sampling_rate)
    return y, sr



def audio_to_melspectrogram(conf, audio):
    spectrogram = librosa.feature.melspectrogram(audio, 
                                                 sr=conf.sampling_rate,
                                                 n_mels=conf.n_mels,
                                                 hop_length=conf.hop_length,
                                                 n_fft=conf.n_fft,
                                                 fmin=conf.fmin,
                                                 fmax=conf.fmax)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram

def mono_to_color(X, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V

def read_as_melspectrogram(conf, pathname, trim_long_data, debug_display=False):
    x = read_audio(conf, pathname, trim_long_data)
    mels = audio_to_melspectrogram(conf, x)
    if debug_display:
        IPython.display.display(IPython.display.Audio(x, rate=conf.sampling_rate))
        show_melspectrogram(conf, mels)
    return mels

def convert_audio_image(wav_file_name):
    # Full audio
    x, sr = read_audio(conf, wav_file_name, trim_long_data=False)
#     print(len(x),x)
    # Split it
    n_parts = math.ceil(len(x) / conf.samples)
#     print('n_parts', n_parts)
    # Pad it
    x = np.append(x, np.zeros(n_parts * conf.samples - len(x)))
#     print('after padding')
#     print(len(x), n_parts * conf.samples) 
    
    # Split it
    splits = [x[i*conf.samples:(i+1)*conf.samples] for i in range(n_parts)]
#     print([len(split) for split in splits])
   
    
    files = [] 
    
    # Save wav & png files
    fname_base = wav_file_name.split('/')[-1][:-4]
    for i, split in enumerate(splits):
        f_split = './tmp/'+ fname_base + '_' + str(i) + '.wav'
#         print('split ', i, ' : ', split.shape, split.dtype, split[:300])
        # Save wave
        librosa.output.write_wav(f_split, split, sr)
#         print(f_split)
        
        # Get the mel
        mel = audio_to_melspectrogram(conf, split)
        color_mel = mono_to_color(mel)
        img = Image.fromarray(color_mel)
        f_img = f_split[:-4]+'.png'
        img.save(f_img)
        
        # append the files
        files.append((f_split, f_img))
    return files
    
    
#     # doing chunking here
#     i = 0
#     csv_files = []
#     img_files = []
#     while len(x) >= conf.samples:
#         chunk = x[0:0+conf.samples]
#         x = x[conf.samples:]
#         x_color = mono_to_color(chunk)
#         img = Image.fromarray(x_color)
#         source = wav_file_name.split('/')[-1]
#         name = source.split('.')[0]
#         img.save('./tmp/' + name+'_'+str(i) +'.png')
#         img_files.append(source)
# #         csv_files.append(name+'_'+str(i)+'.wav')
#         i += 1
#     if len(x) > 0: # pad blank
#         padding = conf.samples - len(x)    # add padding at both ends
#         offset = padding // 2
#         x = np.pad(x, (offset, conf.samples - len(x) - offset), conf.padmode)
#         x_color = mono_to_color(x)
#         img = Image.fromarray(x_color)
#         source = wav_file_name.split('/')[-1]
#         name = source.split('.')[0]
#         img.save('./tmp/' + name+'_'+str(i) +'.png')
#         img_files.append(source)
# #         csv_files.append(name+'_'+str(i)+'.wav')
#     return csv_files, img_files
    
    