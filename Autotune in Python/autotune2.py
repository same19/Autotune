import wave
import numpy as np
import matplotlib.pyplot as plt
import os
import pydub
import IPython.display as ipd
import scipy
from scipy import signal
import math

input = "test_file_1.wav"
# input = "500HzTone.wav"
# input = "Untitled.wav"
output = "output.wav"

fr = 10

def plotNormal(arr):
    plt.figure(figsize=(10,5))
    for i in arr:
        plt.plot(np.linspace(0,100,len(i)),i)
    plt.show()


def plot(plotArray, sr, size = 1, f_ratio = 1):
    plt.figure(figsize=(10,5))
    frequency = np.linspace(0,sr,len(plotArray))
    num_frequency_bins = int(len(frequency) * f_ratio)
    plt.plot(frequency[:(int(num_frequency_bins * size))], np.multiply(plotArray[:(int(num_frequency_bins * size))],1000))
    plt.xlabel("Frequency (Hz)")
    plt.show()

def shift(input, ft, sr):
    return int(input * len(ft) / sr * 0.5)

def frequency(ft, sr):
    # peaks = scipy.signal.find_peaks(np.abs(ft))
    max = np.argmax(np.abs(ft))
    # print(max*sr/len(ft))
    return max*sr/len(ft)


    # truePeaks = []
    # for i in peaks:
    #     if i 
    # print(peaks)

def shiftSound(win, transform, wout, frin):
    sz = win.getframerate()//frin  # Read and process 1/fr second at a time.
    # A larger number for fr means less reverb.
    c = int(win.getnframes()/sz)  # count of the whole file
    # shift = hz//fr  # shifting 100 Hz
    track = np.zeros(c+1)
    for num in range(c):
        da = np.frombuffer(win.readframes(sz), dtype=np.int16)
        left, right = da[0::2], da[1::2]  # left and right channel
        lf, rf = np.fft.rfft(left), np.fft.rfft(right)

        newLf, newRf = np.zeros_like(lf), np.zeros_like(rf)
        f = frequency(lf, win.getframerate())
        # print(f)
        if f == 0:
            current = 1
        else:
            current = (math.log2(f/440)*12) % 1
        # print(f * 2 ** ((1-current)/12))
        # print(current)
        if current > 0.5:
            multiplier = 2 ** ((1-current)/12)
        else:
            multiplier = 2 ** (-current/12)
        multiplier = 2 ** (1/12)

        for i in range(len(newLf)):
            if (int(i/multiplier) < len(lf)):
                newLf[i] = lf[int(i/multiplier)]
        for i in range(len(newRf)):
            if (int(i/multiplier) < len(rf)):
                newRf[i] = rf[int(i/multiplier)]

        nl, nr = np.fft.irfft(newLf), np.fft.irfft(newRf)

        ns = np.column_stack((nl, nr)).ravel().astype(np.int16)
        wout.writeframes(ns.tobytes())
        # plot(np.abs(lf), win.getframerate())
        if f < 1000:
            track[num] = f #* 2 ** ((1-current)/12)
        else:
            track[num] = -100
    
    return track

tracks = []

for i in range(1,12):
    wr = wave.open(input, 'r')
    # Set the parameters for the output file.
    par = list(wr.getparams())
    par[3] = 0  # The number of samples will be set by writeframes.
    par = tuple(par)
    ww = wave.open(output, 'w')
    ww.setparams(par)
    tracks.append(shiftSound(wr, shift, ww, i))
plotNormal(tracks)
avgTrack = [sum(tracks[j][int(i*len(tracks[j])/len(tracks[-1]))] for j in range(len(tracks))) for i in range(len(tracks[-1]))]
plotNormal([avgTrack])
wr.close()
ww.close()