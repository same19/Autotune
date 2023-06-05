# input = "test_file_1.wav"
# input = "500HzTone.wav"
# input = "waning_note.wav"
# input = "one_more_night.wav"
# input = "1-5-1-sound.wav"
# input = "Untitled2.wav"
# input = "radio_brad.wav"
# input = "stranger_long_track.wav"
input = "sam_radio.m4a"
# input = "slidescale.m4a"

mod = 12.0
bias = 0
editsPerSecond = 10

output = "output.wav"

import wave
import numpy as np
import matplotlib.pyplot as plt
import os
import pydub
import IPython.display as ipd
import scipy
from scipy import signal
import math
from pydub import AudioSegment

def openSingleFile(f_in):
    if f_in[-4:] == ".wav":
        wr = wave.open("wav_input/"+f_in, 'r')
    else:
        wrm4a = AudioSegment.from_file("m4a_input/"+f_in,  format= 'm4a')
        file_handle = wrm4a.export("wav_input/"+f_in[:-4]+".wav", format='wav')
        wr = wave.open("wav_input/"+f_in[:-4]+".wav",'r')
    return wr

def open(f_in, f_out):
    if f_in[-4:] == ".wav":
        wr = wave.open("wav_input/"+f_in, 'r')
    else:
        wrm4a = AudioSegment.from_file("m4a_input/"+f_in,  format= 'm4a')
        file_handle = wrm4a.export("wav_input/"+f_in[:-4]+".wav", format='wav')
        wr = wave.open("wav_input/"+f_in[:-4]+".wav",'r')
    # Set the parameters for the output file.
    par = list(wr.getparams())
    par[3] = 0  # The number of samples will be set by writeframes.
    par = tuple(par)
    ww = wave.open("output/"+f_out, 'w')
    ww.setparams(par)
    return wr,ww

fr = 10

def smooth(arr, n):
    out = []
    for i in range(n,len(arr)-n):
        s = 0
        weightsum = 0
        for j in range(-1*n+i,n+1+i):
            slope = (arr[2*i-j]-arr[j])/(2*j+1)
            if slope == 0 or j == 0:
                weight = 10
            else:
                weight = 1/(abs(j) * abs(slope)**2)
            s += arr[j] * weight
            weightsum += weight
        out.append(s/weightsum)
    return out
    # return [sum(arr[j] for j in range(-1*n+i,n+1+i))/(n*2+1) for i in range(n,len(arr)-n)]

def plotNormal(arr,scale=100):
    plt.figure(figsize=(10,5))
    for i in arr:
        plt.plot(np.linspace(0,scale,len(i)),i)
    plt.show()

def plotDict(arr):
    plt.figure(figsize=(10,5))
    plt.plot(arr.keys(), arr.values())
    plt.xlabel("Frequency (Hz)")
    plt.show()
# def plotTuple(arr):


def plot(plotArray, sr, size = 1, f_ratio = 1):
    plt.figure(figsize=(10,5))
    frequency = np.linspace(0,sr,len(plotArray))
    num_frequency_bins = int(len(frequency) * f_ratio)
    plt.plot(frequency[:(int(num_frequency_bins * size))], np.multiply(plotArray[:(int(num_frequency_bins * size))],1))
    plt.xlabel("Frequency (Hz)")
    plt.show()

def shift(input, ft, sr):
    return int(input * len(ft) / sr * 0.5)

def frequencyRaw(ft,sr,low=20,high=1000):
    peaks = scipy.signal.find_peaks(np.abs(ft),height=1*(10**6),distance = int(100.0 * len(ft) / sr+1))[0]
    # print(int(200.0 * len(ft) / sr))
    # print(peaks)
    # plot(ft,sr,0.1)
    # maxIndex = np.argmax(np.abs(ft))
    if len(peaks)>0 and len(peaks < 20) and int(float(peaks[0])*sr/len(ft)) < 1500:
        maxIndex = peaks[0]
    else:
        maxIndex = 0
        return maxIndex,0
    # for i in peaks:
    #     if np.abs(ft)[i] > np.abs(ft)[maxIndex]:
    #         maxIndex = i
    return int(float(maxIndex)*sr/len(ft)),ft[maxIndex]

def frequency(ft, sr, low=20, high=1000):
    peaks = scipy.signal.find_peaks(np.abs(ft),height=1*(10**6),distance = int(100.0 * len(ft) / sr+1))[0]
    # print(int(200.0 * len(ft) / sr))
    # print(peaks)
    # plot(ft,sr,0.1)
    # maxIndex = np.argmax(np.abs(ft))
    if len(peaks)>0 and len(peaks < 20) and int(float(peaks[0])*sr/len(ft)) < 1500:
        maxIndex = peaks[0]
    else:
        maxIndex = 0
        return maxIndex
    # for i in peaks:
    #     if np.abs(ft)[i] > np.abs(ft)[maxIndex]:
    #         maxIndex = i
    return int(float(maxIndex)*sr/len(ft))


    # truePeaks = []
    # for i in peaks:
    #     if i 
    # print(peaks)

def freqAtTime(win, da, time, frequencyRepeats = 50,maxFrameWidth = 1500):
    print(time)
    centerFrame = int(time * win.getframerate()/2)
    # centerFrame = time
    # da = np.frombuffer(win.readframes(win.getnframes()),dtype = np.int16)
    left,right = da[0::2],da[1::2]
    a = centerFrame > len(left)
    # print(str(len(left))+"   " + str(win.getnframes()))
    start = 500
    # maxFrameWidth = 1500
    L = []
    R = []
    num = frequencyRepeats
    for j in range(num):
        i = int(j*(maxFrameWidth-start)/num + start)
        # print(i)
        if centerFrame-i >= 0 and centerFrame+i < len(left):
            lf,rf = np.fft.rfft(left[centerFrame-i:centerFrame+i]),np.fft.rfft(right[centerFrame-i:centerFrame+i])
            # peaks = scipy.signal.find_peaks(np.abs(lf))[0]
            # dubfft = np.fft.rfft(peaks)
            # print(peaks)
            # plot(np.abs(lf),win.getframerate())
            # plotNormal([dubfft],win.getframerate())
            # LF, RF = frequency(lf,win.getframerate()), frequency(rf,win.getframerate())
            # L.append((LF,np.abs(lf[int(LF * len(lf) / win.getframerate())])))
            # R.append((RF,np.abs(rf[int(RF * len(rf) / win.getframerate())])))
            LF,LFmag = frequencyRaw(lf,win.getframerate())
            RF,RFmag = frequencyRaw(rf,win.getframerate())
            L.append((LF,np.abs(LFmag)))
            R.append((RF,np.abs(RFmag)))
        # else:
            # print("NOT IN IF")
            # print(str(centerFrame + i) + " > " + str(len(left)))
    # print(L)
    if len(L) < 1:
        return math.nan
    # print("     " + str(np.argmax(L)))
    # print(np.argmax(L))
    # return np.argmax(L)
    maxElem = L[0]
    maxIndex = start
    for i in range(len(L)):
        width = int(i*(maxFrameWidth-start)/num + start)
        power = 0
        if L[i][1]/(width**power) > maxElem[1]/(maxIndex**power):
            maxElem = L[i]
            maxIndex = width
    return maxElem[0]
    # plotTuple(L)


def getFrequencies(win, tps = 20, frequencyRepeats = 50):  #tps = tones per second measured
    time = win.getnframes()/win.getframerate()
    num = int(tps * time)
    freq = []
    win = openSingleFile(input)
    da = np.frombuffer(win.readframes(win.getnframes()),dtype = np.int16)
    for i in range(num):
        timeInLoop = time * i/num
        freq.append(freqAtTime(win, da, timeInLoop,frequencyRepeats,1500))
    return freq
        


def shiftSound(win, freq, transform, wout, frin):
    #interest is 2.43 seconds and 6.50 seconds
    sz = int(win.getframerate()//frin)  # Read and process 1/fr second at a time.
    # A larger number for fr means less reverb.
    c = int(win.getnframes()/sz)  # count of the whole file
    # shift = hz//fr  # shifting 100 Hz
    track = np.zeros(c+1)
    vol = np.zeros(c+1)
    for num in range(c):
        da = np.frombuffer(win.readframes(sz), dtype=np.int16)
        left, right = da[0::2], da[1::2]  # left and right channel
        lf, rf = np.fft.rfft(left), np.fft.rfft(right)
        # print()
        # print(len(lf))
        # print(len(rf))
        if len(lf) == len(rf) + 1:
            lf = lf[:-1]
        lf2, rf2 = np.fft.fft(lf), np.fft.fft(rf)
        # if num * sz / win.getframerate() - 6.50 < 2/fr:
        #     plot(lf,win.getframerate(),0.1)
        newLf, newRf = np.zeros_like(lf), np.zeros_like(rf)
        if transform:
            f = freq(num/float(c))
        else:
            f = freq(lf, win.getframerate())
        if math.isnan(f):
            print("nan!!!!")
            f = 0
        if f == 0:
            current = 1
        else:
            current = (math.log2(f/440)*mod + bias) % 1
        # print(f * 2 ** ((1-current)/12))
        # print(current)
        if current > 0.5:
            multiplier = 2 ** ((1-current)/mod)
        else:
            multiplier = 2 ** (-current/mod)
        # multiplier = 2 ** (0.5/12)
        if (f > 1000):
            multiplier = 1
        for i in range(len(newLf)):
            if (int(i/multiplier) < len(lf)):
                newLf[i] = lf[int(i/multiplier)]
        for i in range(len(newRf)):
            if (int(i/multiplier) < len(rf)):
                newRf[i] = rf[int(i/multiplier)]
        nl, nr = np.fft.irfft(newLf), np.fft.irfft(newRf)
        # print(len(nl))
        # print(len(nr))
        ns = np.column_stack((nl, nr)).ravel().astype(np.int16)
        wout.writeframes(ns.tobytes())
        # plot(np.abs(lf), win.getframerate())
        track[num] = f
        # if (f > 5000):
        #     plot(lf, win.getframerate())
        # vol[num] = sum(i for i in left)/len(left)
        vol[num] = np.abs(lf[int(f / win.getframerate() * len(lf))] / 1000)

        # print(num)
    return track,vol

tracks = []
vols = []
start = 15
stop = 15
num = 10
for j in range(num):
    i = (j/float(num)) * (stop-start) + start
    wr, ww = open(input,output)
    # plotNormal([getFrequencies(wr)])
    # exit(0)
    nt,nv = shiftSound(wr, frequency, False, ww, i)
    tracks.append(nt)
    vols.append(nv)
# plotNormal(tracks)
time = wr.getnframes()/wr.getframerate()
# plotNormal(vols,time)

# for i in range(len(tracks[0])):
#     nancount = 0
#     for j in range(len(tracks)):
#         if (math.isnan(tracks[j][i])):
#             nancount += 1
#             if nancount >= (len(tracks[0])//3):
#                 for j in range(len(tracks)): #set this spot to nan
#                     tracks[j][i] = math.nan
#                 break
#             if j==0:
#                 tracks[j][i] = 0
#             else:
#                 tracks[j][i] = tracks[j-1][i]
avgTrack = [sum(tracks[j][int(i*len(tracks[j])/len(tracks[-1]))] for j in range(len(tracks)))/len(tracks) for i in range(len(tracks[-1]))]
avgVol = [abs(sum(vols[j][int(i*len(vols[j])/len(vols[-1]))] for j in range(len(vols)))/len(vols)) for i in range(len(vols[-1]))]

n = int(0.35*wr.getframerate() * len(avgTrack)/float(wr.getnframes())) #moving average - amount around in each direction
smoothedTrack = smooth(avgTrack,n)
# smoothedTrack = smooth(smoothedTrack,n)
n = int(0.08*wr.getframerate() * len(avgTrack)/float(wr.getnframes())) #moving average - amount around in each direction
smoothedTrack = smooth(smoothedTrack,n)
smoothedTrack = smooth(smoothedTrack,n)
smoothedTrack = smooth(smoothedTrack,n)
smoothedTrack = smooth(smoothedTrack,n)
n = int(0.1*wr.getframerate() * len(avgTrack)/float(wr.getnframes()))
smoothedVol = smooth(avgVol,n)

# plotNormal([avgVol])
# print((0.25/time)*len(smoothedTrack))
peaks = scipy.signal.find_peaks(np.multiply(avgTrack,-1),threshold=50,width=(0,(0.3/time)*len(avgTrack)))[0]
# print(peaks[3])
wr,ww = open(input,output)
# freqAtTime(wr,peaks[0] / len(avgTrack) * time)
# plotNormal(tracks,time)
# plotNormal([avgTrack,avgVol],time)
#0.25 seconds = (0.25/time)*len(smoothedTrack)
# plotNormal([smoothedTrack,avgVol],time)
# plotNormal([smoothedVol],time)
print("graphing")
ACTUAL_FREQUENCIES = getFrequencies(wr)
plotNormal([avgTrack,ACTUAL_FREQUENCIES],time)

def getProcessedFrequency(fracThrough):
    return ACTUAL_FREQUENCIES[int(fracThrough * len(ACTUAL_FREQUENCIES))]
wr,ww = open(input,output)
shiftSound(wr, getProcessedFrequency,True,ww,editsPerSecond)
wr.close()
ww.close()