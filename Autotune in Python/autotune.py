import numpy as np
import matplotlib.pyplot as plt
import os
import pydub
import IPython.display as ipd
print()
def rollCut(arr, num, fill_value=0):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result
def read(f, normalized=False):
    """MP3 to numpy array"""
    a = pydub.AudioSegment.from_mp3(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2**15
    else:
        return a.frame_rate, y

def write(f, sr, x, normalized=False):
    """numpy array to MP3"""
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x)
    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(f, format="mp3", bitrate="320k")

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

# audio_file = "test_file_1.wav"
audio_file = "500HzTone.wav"

sr, arr = read(audio_file)
freqShift = 200


def plot_magnitude_spectrum(ftMagnitudeHere, size = 1, f_ratio = 1):
    plt.figure(figsize=(10,5))
    frequency = np.linspace(0,sr,len(ftMagnitudeHere))
    num_frequency_bins = int(len(frequency) * f_ratio)
    plt.plot(frequency[:(int(num_frequency_bins * size))], ftMagnitudeHere[:(int(num_frequency_bins * size))])
    plt.xlabel("Frequency (Hz)")
    plt.show()

def editSound(arr, fourierEditFunc, sr):
    arr1 = arr
    while len(arr1.shape)>1:
        arr1 = arr1[:,0]
    ft = np.fft.fft(arr1)
    print(np.argmax(ft))
    # ftMagnitude = np.abs(ft)
    # plot_magnitude_spectrum(ftMagnitude, 0.03)
    # plot_magnitude_spectrum(np.abs(ft), 0.04)
    rolled = np.roll(ft, 2000)
    print(np.argmax(rolled))
    # plot_magnitude_spectrum(np.abs(rolled), 0.04)
    plot_magnitude_spectrum(np.fft.ifft(rolled))
    write("output.wav", sr, np.fft.ifft(rolled))
    transformedft = fourierEditFunc(ft, ftMagnitude, sr)
    transformed = np.fft.ifft(transformedft)
    return transformed
def fourierEditFunc1(ft, ftMagnitude, sr):
    space = sr/len(ftMagnitude)
    transformedft = np.concatenate((
    rollCut(ft[:len(ft)//2],int(freqShift/space)),
    rollCut(ft[len(ft)//2:],-int(freqShift/space)) 
    ))
    return transformedft
plot_magnitude_spectrum(arr)
reversed = editSound(arr, fourierEditFunc1, 10)
# plot_magnitude_spectrum(reversed)
write("output.wav", sr, reversed)


