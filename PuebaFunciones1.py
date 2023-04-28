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

WAVEFORM_path_export = 'waveform'
SPECTROGRAM_path_export='spectogram'
GREYSPECTROGRAM_path_export='grey spectrogram'
MELSPECTROGRAM_path_export='mel spectrogram'
CHROMAGRAM_path_export='chromagram'
MFCC_path_export='mfcc'
DELTA_MFCC_path_export='delta mfccs'
DELTA2_MFCC_path_export='delta2 mfccs'
FvsA_path_export='FrequencyAmplitude'
AMPLITUDEENV_path_export='amplitude envelope'
RMSE_path_export='root mean square energy'
ZCR_path_export='zero croosing rate'
BER_path_export='band energy ratio'
SpecCent_path_export='spectral centroid'
Bandwidth_path_export='bandwidth'
clip = (r'C:\Users\BHC4SLP\Documents\Python Projects\Proyecto2-GraficaAudio\PruebaAudio1.wav')

def LoadAudio_Turn2Decibels(clip):
    y, sr = librosa.load(clip) 
    D = librosa.stft(y) 
    # STFT of y 
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max) 
    #, ref=np.max

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

def calculate_split_frequency_bin(split_frequency, sample_rate, num_frequency_bins):
    """Infer the frequency bin associated to a given split frequency."""
    
    frequency_range = sample_rate / 2
    frequency_delta_per_bin = frequency_range / num_frequency_bins
    split_frequency_bin = math.floor(split_frequency / frequency_delta_per_bin)
    return int(split_frequency_bin)



def band_energy_ratio(spectrogram, split_frequency, sample_rate):
    """Calculate band energy ratio with a given split frequency."""
    
    split_frequency_bin = calculate_split_frequency_bin(split_frequency, sample_rate, len(spectrogram[0]))
    band_energy_ratio = []
    
    # calculate power spectrogram
    power_spectrogram = np.abs(spectrogram) ** 2
    power_spectrogram = power_spectrogram.T
    
    # calculate BER value for each frame
    for frame in power_spectrogram:
        sum_power_low_frequencies = frame[:split_frequency_bin].sum()
        sum_power_high_frequencies = frame[split_frequency_bin:].sum()
        band_energy_ratio_current_frame = sum_power_low_frequencies / sum_power_high_frequencies
        band_energy_ratio.append(band_energy_ratio_current_frame)
    
    return np.array(band_energy_ratio)

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
plt.figure(figsize=(25, 10))
fig, ax = plt.subplots() 
img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear') 
img = librosa.display.specshow(S_db, x_axis='time', y_axis='log') 
plt.colorbar(format="%+2.f")
ax.set(title='SPECTROGRAM') 
guardarimagen(SPECTROGRAM_path_export,'Spectrogram',fig)
plt.close()

"""Grey Spectrogram"""
# SPECTROGRAM representation - object-oriented interface 
fig, ax = plt.subplots() 
img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear') 
img = librosa.display.specshow(S_db, x_axis='time', y_axis='log', cmap='gray_r') 
plt.colorbar(format="%+2.f")
ax.set(title='GREY SPECTROGRAM') 
guardarimagen(GREYSPECTROGRAM_path_export,'Grey Spectrogram',fig)
plt.close()

"""Mel Spectrogram"""
#Extracting Mel Spectrogram
mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=90)
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

"""Chromogram"""
#CHROMAGRAM representation - object-oriented interface 
CHROMAGRAM = librosa.feature.chroma_cqt(y=y, sr=sr) 
fig, ax = plt.subplots() 
img = librosa.display.specshow(CHROMAGRAM, y_axis='chroma', x_axis='time') 
plt.colorbar(format="%+2.f")
ax.set(title='CHROMAGRAM') 
guardarimagen(CHROMAGRAM_path_export,'Chromogram',fig)
plt.close()

"""MFCCs"""
#MFCC representation - object-oriented interface 
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=1200) 
fig, ax = plt.subplots() 
img = librosa.display.specshow(mfccs, x_axis='time') 
plt.colorbar(format="%+2.f")
ax.set(title='Mel-frequency cepstral coefficients (MFCCs)') 
guardarimagen(MFCC_path_export,'MFCCs',fig)
plt.close()

"""Delta MFCCs"""
delta_mfccs = librosa.feature.delta(mfccs)

plt.figure(figsize=(25, 10))
fig, ax = plt.subplots() 
img = librosa.display.specshow(delta_mfccs, x_axis='time',sr=sr) 
plt.colorbar(format="%+2.f")
ax.set(title='Delta Mel-frequency cepstral coefficients (MFCCs)') 
guardarimagen(DELTA_MFCC_path_export,'MFCCs',fig)
plt.close()

"""Delta2 MFCCs"""
delta2_mfccs = librosa.feature.delta(mfccs, order=2)

plt.figure(figsize=(25, 10))
fig, ax = plt.subplots() 
img = librosa.display.specshow(delta2_mfccs, x_axis='time',sr=sr) 
plt.colorbar(format="%+2.f")
ax.set(title='Delta2 Mel-frequency cepstral coefficients (MFCCs)') 
guardarimagen(DELTA2_MFCC_path_export,'MFCCs',fig)
plt.close()

"""Band Energy Ratio"""
HOP_SIZE=512
y_spec = librosa.stft(y, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)

split_frequency_bin = calculate_split_frequency_bin(2000, 22050, 1025)
split_frequency_bin

ber_y = band_energy_ratio(y_spec, 2000, sr)
len(ber_y)
#Visualise Band Energy Ratio
frames = range(len(ber_y))
t = librosa.frames_to_time(frames, hop_length=HOP_SIZE)

plt.figure(figsize=(25, 10))
fig, ax = plt.subplots()
plt.plot(t, ber_y, color="b")
#plt.ylim((0, 200))
ax.set(title="Band Energy Ratio")
guardarimagen(BER_path_export,'Band Energy Ratio',fig)
plt.close()

"""Spectral Centroid"""
sc_y = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
sc_y.shape

#Visualising spectral centroid
len(t)
plt.figure(figsize=(25,10))
fig, ax = plt.subplots()
plt.plot(t, sc_y, color='b')
ax.set(title="Spectral Centroid")
guardarimagen(SpecCent_path_export,'Spectral Centroid',fig)
plt.close()

"""Bandwidth"""
#Spectral bandwidth with Librosa
ban_y = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
ban_y.shape
#Visualising spectral bandwidth
plt.figure(figsize=(25,10))
fig, ax = plt.subplots()
plt.plot(t, ban_y, color='b')
ax.set(title="Bandwidth")
guardarimagen(Bandwidth_path_export,'Bandwidth',fig)
plt.close()