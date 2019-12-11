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
    
path_load = './original_data/71_merge.mp4'
path_save = './new_data/71_merge.wav'
transform(path_load, path_save, on = False)

sr, audio_data = read_audio_data(path_save)
sr_new = 16000
audio_data = mono(audio_data, sr, sr_new)
FourierTransformation = np.abs(sp.fft(audio_data))

order = 10
cutoff = [50,6000]
# Butterworth filter
sos = signal.butter(order, cutoff, 'bp', fs=sr_new, output='sos')
filtered_audio = signal.sosfilt(sos, audio_data)

#plot_spec(audio_data, sr_new, int(0.05*sr_new), int(0.05*sr_new))

#%%
path_write = './new_data/71_merge_filtered.wav'
write_audio_data(path_write, sr_new, filtered_audio)


