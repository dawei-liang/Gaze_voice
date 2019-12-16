#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 13:59:58 2019

@author: dawei
"""

from scipy.io import wavfile
import numpy as np
import os
from python_speech_features import mfcc
import pandas as pd


#%%
def read_audio_data(file):
    """
    read audio, only support 16-bit depth
    """
    rate, wav_data = wavfile.read(file)
    assert wav_data.dtype == np.int16, 'Not support: %r' % wav_data.dtype  # check input audio rate(int16)
    scaled_data = wav_data / 32768.0   # 16bit standardization
    return rate, scaled_data


#%%
positive_path = './segmented_data/person/'
negative_path = './segmented_data/background/'

positive_list = [x for x in os.listdir(positive_path) if x.endswith('.wav')]
negative_list = [x for x in os.listdir(negative_path) if x.endswith('.wav')] 

window = (1/75)   # seconds
hopping = window * 1   # seconds
sr_new = 16050

# Loop for each subject
wav_pos = np.empty((0, 15))
wav_neg = np.empty((0, 15))
for i in positive_list:
    sr, wav = read_audio_data(os.path.join(positive_path, i))
    mfcc_feat = mfcc(wav, sr, window, hopping, 15)
    wav_pos = np.vstack((wav_pos, mfcc_feat))    
for i in negative_list:
    sr, wav = read_audio_data(os.path.join(negative_path, i))
    mfcc_feat = mfcc(wav, sr, window, hopping, 15)
    wav_neg = np.vstack((wav_neg, mfcc_feat))
    
pos_label = np.ones((wav_pos.shape[0], 1))
neg_label = np.zeros((wav_neg.shape[0], 1))

wav_pos_with_label = np.hstack((wav_pos, pos_label))
wav_neg_with_label = np.hstack((wav_neg, neg_label))

pd.DataFrame(wav_pos_with_label).to_csv(positive_path + './pos.csv')
pd.DataFrame(wav_neg_with_label).to_csv(negative_path + './neg.csv')
    

 