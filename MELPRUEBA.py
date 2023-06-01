import librosa
import librosa.display
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
FvsA_path_export='FrequencyAmplitude'
WAVEFORM_path_export='waveform'
clip = (r'C:\Users\BHC4SLP\Documents\Python Projects\Proyecto2-GraficaAudio\redhot.wav')

def guardarimagen(path_export,NombreImag,fig):
    audio_filename = os.path.basename(os.path.normpath(clip)) 
    image_filename_to_save = str(audio_filename).replace(".wav", "-", 1) + NombreImag+".png" 
    if not os.path.exists(path_export): 
        os.makedirs(path_export) 
    fig.savefig(os.path.join(path_export,image_filename_to_save)) 

scale, sr = librosa.load(clip)
""" Waveform"""
# Simple WAVEFORM to check clip trimming accuracy 
fig, ax = plt.subplots() 
img = librosa.display.waveshow(scale, sr=sr) 
ax.set(title='WAVEFORM') 
#The first strips off any trailing slashes, the second gives you the last part of the path. 
guardarimagen(WAVEFORM_path_export,'waveform',fig)
plt.close()

"""fft=np.fft.fft(scale)

magnitude=np.abs(fft)
frequency=np.linspace(0,sr,len(magnitude))

left_frequency=frequency[:int(len(frequency)/2)]
left_magnitude=magnitude[:int(len(frequency)/2)]

fig, bx=plt.subplots()
plt.plot(left_frequency,left_magnitude)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
#plt.show()
bx.set(title="Frequency vs Amplitude")
guardarimagen(FvsA_path_export,'FvsA',fig)
plt.close()"""


#Mel filter banks
"""filter_banks = librosa.filters.mel(n_fft=2048, sr=22050, n_mels=10)
plt.figure(figsize=(25, 10))
librosa.display.specshow(filter_banks, 
                         sr=sr, 
                         x_axis="linear")
plt.colorbar(format="%+2.f")
plt.show()"""

#Extracting Mel Spectrogram
"""mel_spectrogram = librosa.feature.melspectrogram(y=scale, sr=sr, n_fft=2048, hop_length=512, n_mels=90)
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

Spectrogram
# SPECTROGRAM representation - object-oriented interface 
fig, ax = plt.subplots() 
img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax) 
img = librosa.display.specshow(S_db, x_axis='time', y_axis='log', ax=ax) 
ax.set(title='SPECTROGRAM') 
guardarimagen(SPECTROGRAM_path_export,'Spectrogram',fig)
plt.close()"""
 #Frecuencia vs amplitud primer modelo
"""Audio_m=y[:] #Renombramos este arreglo como Audio_m
L=len(Audio_m)
gk=fourier.fft(Audio_m) #Transformada de fourier sobre el vector con los valores del audio
M_gk=abs(gk)            #Calculo de su valor absoluto de los nuevos valores tras la transformada
M_gk=M_gk[0:L//2]       #Funcion par asi que basta con analizar la mitad de los valores
F=(sr/L)*np.arange(0,L//2) #Se declara un arreglo hasta L medios
fig, bx=plt.subplots()
plt.plot(F,M_gk)
plt.xlabel('Frecuencia (Hz)', fontsize='14')
plt.ylabel('Amplitud FFT', fontsize='14')
#plt.show()
bx.set(title="Frequency vs Amplitude")
guardarimagen(FvsA_path_export1,FvsA_path_export2,res,'FvsA',fig)
plt.close()"""