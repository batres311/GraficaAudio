import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
import pyaudio #Libreria que ayuda para obtener el audio y darle formato
import wave  #Permite leer y escribir archivos wav
import winsound #Permite acceder a la maquinaria básica de reproducción de sonidos proporcionada por la plataformas Windows.
import scipy.io.wavfile as waves #libreria importante para los datos del audio
import scipy.fftpack as fourier #libreria para pasar al dominio de la frecuencia de forma sencilla

WAVEFORM_path_export = 'waveform'
SPECTROGRAM_path_export='spectogram'
CHROMAGRAM_path_export='chromagram'
MFCC_path_export='mfcc'
FvsA_path_export='FrequencyAmplitude'
AMPLITUDEENV_path_export='amplitude envelope'
RMSE_path_export='root mean square energy'
ZCR_path_export='zero croosing rate'
clip = (r'C:\Users\BHC4SLP\Documents\Python Projects\Proyecto2-GraficaAudio\PruebaAudio1.wav')

def LoadAudio_Turn2Decibels(clip):
    y, sr = librosa.load(clip) 
    D = librosa.stft(y) 
    # STFT of y 
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max) 

    return y,S_db,sr

def guardarimagen(path_export,NombreImag,fig):
    audio_filename = os.path.basename(os.path.normpath(clip)) 
    image_filename_to_save = str(audio_filename).replace(".wav", "-", 1) + NombreImag+".png" 
    if not os.path.exists(path_export): 
        os.makedirs(path_export) 
    fig.savefig(os.path.join(path_export,image_filename_to_save)) 

def fancy_amplitude_envelope(signal, frame_size, hop_length):
    """Fancier Python code to calculate the amplitude envelope of a signal with a given frame size."""
    return np.array([max(signal[i:i+frame_size]) for i in range(0, len(signal), hop_length)])

y,S_db,sr=LoadAudio_Turn2Decibels(clip)

""" Waveform"""
# Simple WAVEFORM to check clip trimming accuracy 
fig, ax = plt.subplots() 
img = librosa.display.waveshow(y, sr=sr, axis='time') 
ax.set(title='WAVEFORM') 
#The first strips off any trailing slashes, the second gives you the last part of the path. 
guardarimagen(WAVEFORM_path_export,'waveform',fig)
plt.close()

"""Amplitude envelope"""
FRAME_SIZE = 1024
HOP_LENGTH = 512

# number of frames in amplitude envelope
ae_y = fancy_amplitude_envelope(y, FRAME_SIZE, HOP_LENGTH)
len(ae_y)

#Visualizing amplitud envelope
frames = range(len(ae_y))
t = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)

fig, ax = plt.subplots()
img=librosa.display.waveshow(y, alpha=0.5)
plt.plot(t, ae_y, color="r")
#plt.ylim((-1, 1))
ax.set(title="Amplitude envelope")
guardarimagen(AMPLITUDEENV_path_export,'AmplitudEnvelope',fig)
plt.close()

"""Root-mean-squared energy with Librosa"""
rms_y = librosa.feature.rms(y=y, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
#Visualise RMSE + waveform
frames = range(len(rms_y))
t = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)
# rms energy is graphed in red
plt.figure(figsize=(15, 17))
fig, ax = plt.subplots()
librosa.display.waveshow(y, alpha=0.5)
plt.plot(t, rms_y, color="r")
#plt.ylim((-1, 1))
ax.set(title="RMS energy")
guardarimagen(RMSE_path_export,'RMSE',fig)
plt.close()

""" Zero crossing rate"""
#Zero-crossing rate with Librosa
zcr_y = librosa.feature.zero_crossing_rate(y, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
zcr_y.size

#Visualise zero-crossing rate with Librosa
plt.figure(figsize=(15, 10))
fig, ax = plt.subplots()
plt.plot(t, zcr_y, color="r")
plt.ylim(0, 1)
#plt.show()
ax.set(title="Zero Croosing Rate")
guardarimagen(ZCR_path_export,'ZCR',fig)
plt.close()

"""Frequency vs Amplitude"""
#Frequency vs amplitude graph
Audio_m=y[:] #Renombramos este arreglo como Audio_m
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
guardarimagen(FvsA_path_export,'FvsA',fig)
plt.close()

"""Spectrogram"""
# SPECTROGRAM representation - object-oriented interface 
fig, ax = plt.subplots() 
img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax) 
img = librosa.display.specshow(S_db, x_axis='time', y_axis='log', ax=ax, cmap='gray_r') 
ax.set(title='SPECTROGRAM') 
guardarimagen(SPECTROGRAM_path_export,'Spectrogram',fig)
plt.close()

"""Chromogram"""
#CHROMAGRAM representation - object-oriented interface 
CHROMAGRAM = librosa.feature.chroma_cqt(y=y, sr=sr) 
fig, ax = plt.subplots() 
img = librosa.display.specshow(CHROMAGRAM, y_axis='chroma', x_axis='time', ax=ax) 
ax.set(title='CHROMAGRAM') 
guardarimagen(CHROMAGRAM_path_export,'Chromogram',fig)
plt.close()

"""MFCCs"""
#MFCC representation - object-oriented interface 
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=1200) 
fig, ax = plt.subplots() 
img = librosa.display.specshow(mfccs, x_axis='time') 
ax.set(title='Mel-frequency cepstral coefficients (MFCCs)') 
guardarimagen(MFCC_path_export,'MFCCs',fig)
plt.close()