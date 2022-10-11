#select .wav file, read and plot
#calculate Fourier Transform and display absolute value 0-500 Hz
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt
Tk().withdraw()
filename = askopenfilename() #open file dialog

# read audio samples
input_data = read(test-string.wav)
signal = input_data[1]
sampling_freq = input_data[0]
time = np.arange(len(signal))/sampling_freq

def plot_data(start_time,end_time):
	#function to plot data between start_time and end_time

	time_index1 = time.tolist().index(start_time)
	time_index2 = time.tolist().index(end_time)
	plt.figure()
	plt.plot(time[time_index1:time_index2+1],signal[time_index1:time_index2+1])
	plt.ylabel("Amplitude [a.u.]")
	plt.xlabel("Time (s)")
	plt.title("Recorded Signal")
	plt.show()

def FT_data(data,sampling_rate):
 #function to calcuate and display absolute value of Fourier Transform

	freq = 0.5 * sampling_rate * np.linspace(-1.0, 1.0, len(data))
	FTdata = np.fft.fftshift(np.fft.fft(np.fft.fftshift(data)))

	freq_index1 = np.amin(np.where(freq >= 0))
	freq_index2 = np.amin(np.where(freq >= 500))
	plt.figure()
	plt.plot(freq[freq_index1:freq_index2+1],abs(FTdata[freq_index1:freq_index2+1]))
	plt.ylabel("Magnitude [a.u.]")
	plt.xlabel("Frequency (Hz)")
	plt.title("Absolute Value of Fourier Transform")
	plt.show()

plot_data(0,0.25) #plot signal in time window defined by 2 values
FT_data(signal,sampling_freq) #Fourier Transfomr and plot absolute value