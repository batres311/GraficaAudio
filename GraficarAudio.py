# Importar librerias
import matplotlib.pyplot as plt
import numpy as np
import librosa

file=(r'C:\Users\BHC4SLP\Documents\Python Projects\Proyecto2-GraficaAudio\PruebaAudio1.wav')
data, sr=librosa.load(file)

data, _=librosa.effects.trim(data)

n=1024

fourier=np.abs(librosa.stft(data[:n]))
plt.plot(fourier)
plt.show()
