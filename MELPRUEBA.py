import librosa
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import pyaudio #Libreria que ayuda para obtener el audio y darle formato
import wave  #Permite leer y escribir archivos wav
import winsound #Permite acceder a la maquinaria básica de reproducción de sonidos proporcionada por la plataformas Windows.
import scipy.io.wavfile as waves #libreria importante para los datos del audio
import scipy.fftpack as fourier #libreria para pasar al dominio de la frecuencia de forma sencilla

MELSPECTROGRAM_path_export='mel spectogram'
clip = (r'C:\Users\BHC4SLP\Documents\Python Projects\Proyecto2-GraficaAudio\redhot.wav')

def guardarimagen(path_export,NombreImag,fig):
    audio_filename = os.path.basename(os.path.normpath(clip)) 
    image_filename_to_save = str(audio_filename).replace(".wav", "-", 1) + NombreImag+".png" 
    if not os.path.exists(path_export): 
        os.makedirs(path_export) 
    fig.savefig(os.path.join(path_export,image_filename_to_save)) 

scale, sr = librosa.load(clip)

#Mel filter banks
"""filter_banks = librosa.filters.mel(n_fft=2048, sr=22050, n_mels=10)
plt.figure(figsize=(25, 10))
librosa.display.specshow(filter_banks, 
                         sr=sr, 
                         x_axis="linear")
plt.colorbar(format="%+2.f")
plt.show()"""

#Extracting Mel Spectrogram
mel_spectrogram = librosa.feature.melspectrogram(y=scale, sr=sr, n_fft=2048, hop_length=512, n_mels=90)
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

plt.figure(figsize=(25, 10))
fig, ax = plt.subplots() 
img=librosa.display.specshow(log_mel_spectrogram, 
                         x_axis="time",
                         y_axis="mel", 
                         sr=sr)
plt.colorbar(format="%+2.f")
ax.set(title='SPECTROGRAM') 
guardarimagen(MELSPECTROGRAM_path_export,'Spectrogram',fig)
plt.close()

"""Spectrogram
# SPECTROGRAM representation - object-oriented interface 
fig, ax = plt.subplots() 
img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax) 
img = librosa.display.specshow(S_db, x_axis='time', y_axis='log', ax=ax) 
ax.set(title='SPECTROGRAM') 
guardarimagen(SPECTROGRAM_path_export,'Spectrogram',fig)
plt.close()"""