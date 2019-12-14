#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:09:18 2019

@author: dawei
"""

import subprocess
import scipy as sp 
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import resampy
from scipy import signal
import pandas as pd
import os

#%%

def transform(path_load, path_save, on = False):
    """
    convert to .wav with 16 bits
    """
    if on:
        # to change volumn:
        # + " -filter:a 'volume=0' " \
        # chop (in seconds): + " -ss 3 -to 21 " \
        command = "ffmpeg -i " + path_load \
                + " -vn -sample_fmt s16 " \
                + " -filter:a 'volume=1' " \
                + path_save
        subprocess.call(command, shell=True)

        
def read_audio_data(file):
    """
    read audio, only support 16-bit depth
    """
    rate, wav_data = wavfile.read(file)
    assert wav_data.dtype == np.int16, 'Not support: %r' % wav_data.dtype  # check input audio rate(int16)
    scaled_data = wav_data / 32768.0   # 16bit standardization
    return rate, scaled_data

def mono(data, sr, sr_new):
    """
    Convert to mono.
    """
    try:
        if data.shape[1] > 1:
            data = np.mean(data, axis=1)
    except:
        pass
    # Resampling the data to specified rate
    if sr != sr_new:
      data = resampy.resample(data, sr, sr_new)
    return data

def write_audio_data(filename, rate, wav_data):
    '''write normalized audio signals with 16 bit depth to a wave file'''
    wav_data = wav_data * 32768.0   # 16bit
    wav_data = wav_data.astype(np.int16)
    wavfile.write(filename, rate, wav_data)
    print(filename + ' Saved')

def framing(data, window_length, hop_length):
    """
    Convert 1D time series signals or N-Dimensional frames into a (N+1)-Dimensional array of frames.
    No zero padding, rounding at the end.
    Args:
        data: Input signals.
        window_length: Number of samples in each frame.
        hop_length: Advance (in samples) between each window.
    Returns:
        np.array with as many rows as there are complete frames that can be extracted.
    """
    
    num_samples = data.shape[0]
    frame_array = data[0:window_length]
    # create a new axis as # of frames
    frame_array = frame_array[np.newaxis]  
    start = hop_length
    for _ in range(num_samples):
        end = start + window_length
        if end <= num_samples:
            # framing at the 1st axis
            frame_temp = data[start:end]
            frame_temp = frame_temp[np.newaxis]
            frame_array = np.concatenate((frame_array, frame_temp), axis=0)
        start += hop_length
    return frame_array


def reconstruct_time_series(frames, hop_length_samples):
    """
    Reconstruct N-Dimensional framed array back to (N-1)-Dimensional frames or 1D time series signals
    Args:
        frames = [# of frames, window length1 in samples, (window length2, ...)]
        hop_length_samples = # of samples skipped between two frames
    return:
        (N-1)-Dimensional frames or 1D time series signals
    """
    new_signal = []
    for i in range(len(frames)-1):
        for j in range(0, hop_length_samples):
            new_signal.append(frames[i, j])
    # Last frame
    for i in range(frames.shape[1]):
        new_signal.append(frames[-1,i])
        
    new_signal = np.asarray(new_signal)
    
    return new_signal


def noise_filter(audio_data, order=10, cutoff=[50, 6000], sr=16000):
    """
    Butterworth filter
    Args: 
        audio_data: raw audio signals
        order: filter order
        cutoff: cutoff freq (1/sqrt(2)), can be float or array of two floats
        sr: sampling rate
    Out:
        filtered audio signals
    """
    sos = signal.butter(order, cutoff, 'bp', fs=sr, output='sos')
    filtered_audio = signal.sosfilt(sos, audio_data)
    return filtered_audio
        
    
def plot_spec(audio_data, sr, window_length, hop_length):
    """
    Plot freq spectrum of input signals
    Args:
        audio_data: Input signals.
        window_length: Number of samples in each frame.
        hop_length: Advance (in samples) between each window.
    """
    
    f_distirbution=[]   # f distribution for each window
    framed_audio = framing(audio_data, window_length, hop_length)
    
    for i in range(len(framed_audio)): 
        k = sp.arange(window_length)
        T = window_length/sr   # time duration of a window
        frq = k/T # two sides frequency range, [0, 1/T, 2/T, ..., sampling rate]
        frq = frq[range(int(window_length/2))] # one side frequency range, [0, 1/T, 2/T, ..., sampling rate/2 (Nqy f)]
        
        Y = sp.fft(framed_audio[i]) / window_length # fft computing and normalization
        Y = abs(Y[range(int(window_length/2))])   # one side freq amplitute
        f_distirbution.append(Y)
    
    for i in range(len(f_distirbution)):
        plt.plot(frq,f_distirbution[i],'r') # plotting the spectrum
    plt.xlabel('Freq (Hz)')
    plt.ylabel('|Y(freq)|')
    plt.show()
            


#%%
order = 10
cutoff = [1000,6000]
window = (1/75)   # seconds
hopping = window * 1   # seconds
sr_new = 16050

#noise_filter(audio_data, order, cutoff, sr_new)
#plot_spec(audio_data, sr_new, int(0.05*sr_new), int(0.05*sr_new))


ref = pd.read_csv('./original_data/360_speech_audio_video/label_gaze_table.csv')
id_list = set(ref.id.values)
path_load = './original_data/360_speech_audio_video/'
video_file_list = [x for x in os.listdir(path_load) if x.endswith('.mp4')]   # Load video files

#%%
# Loop for each subject
for i in id_list:
    for video in video_file_list:
        if str(i) in video:
            target = video
    # transform from mp4 to wav
    path_save = './transformed_data/' + target.replace('.mp4', '') + '.wav'
    print('transforming video %d to wav' %i)
    transform(path_load + target, path_save, on = True)  
    
    # Load wav for segmentation
    print('begin to segment wave file %d' %i)
    sr, audio_data = read_audio_data(path_save)
    audio_data = mono(audio_data, sr, sr_new)
    framed_audio = framing(audio_data, int(window * sr_new), int(hopping * sr_new))
    
    # segmentation
    frames_background, frames_person = np.empty((1, framed_audio.shape[1])), np.empty((1, framed_audio.shape[1]))
    start_background = (ref.loc[(ref['id'] == i) & (ref['roi1_person_yn'] == 'Background'), 'fix_start_frame']) * ((1/75) // hopping) - 1
    end_background = (ref.loc[(ref['id'] == i) & (ref['roi1_person_yn'] == 'Background'), 'fix_end_frame']) * ((1/75) // hopping) - 1
    start_person = (ref.loc[(ref['id'] == i) & (ref['roi1_person_yn'] == 'person'), 'fix_start_frame']) * ((1/75) // hopping) - 1
    end_person = (ref.loc[(ref['id'] == i) & (ref['roi1_person_yn'] == 'person'), 'fix_end_frame']) * ((1/75) // hopping) - 1   
    frames_background_idx = np.vstack((start_background.values, end_background.values)).T
    frames_person_idx = np.vstack((start_person.values, end_person.values)).T
    
    # stack sound segments based on categories
    for j in range(len(frames_background_idx)):  
        frames_background = np.vstack((frames_background, framed_audio[int(frames_background_idx[j,0]) : int(frames_background_idx[j,1])]))
    for k in range(len(frames_person_idx)):        
        frames_person = np.vstack((frames_person, framed_audio[int(frames_person_idx[k,0]) : int(frames_person_idx[k,1])]))
    
    # reconstructed audio signals for each category per subject, idx from 1:end
    audio_background = reconstruct_time_series(frames_background[1:], int(hopping * sr_new))
    audio_person = reconstruct_time_series(frames_person[1:], int(hopping * sr_new))
    # save segmented wav for each subject and each categories seperately
    write_audio_data('./segmented_data/background/' + target.replace('_merge.mp4', '') + '.wav', sr_new, audio_background)
    write_audio_data('./segmented_data/person/' + target.replace('_merge.mp4', '') + '.wav', sr_new, audio_person)



