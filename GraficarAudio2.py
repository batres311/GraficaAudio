import winsound
filename=(r'C:\Users\BHC4SLP\Documents\Python Projects\Proyecto2-GraficaAudio\ejemplo2.wav')
winsound.PlaySound(filename,winsound.SND_FILENAME)

import scipy.io.wavfile as waves
Fs, data=waves.read(filename)
Audio_m=data[:,0]

import numpy as np

L=len(Audio_m)
Ts=0.001
n=Ts*np.arange(0,L)

import matplotlib.pyplot as plt

fig, ax=plt.subplots()
plt.plot(n,Audio_m)
plt.xlabel('Tiempo (s)')
plt.ylabel('Audio')


import scipy.fftpack as fourier
gk=fourier.fft(Audio_m)
M_gk=abs(gk)
M_gk=M_gk[0:L//2]

F=(Fs/L)*np.arange(0,L//2)
fig, bx=plt.subplots()
plt.plot(F,M_gk)
plt.xlabel('Frecuencia (Hz)', fontsize='14')
plt.ylabel('Amplitud FFT', fontsize='14')
plt.show()