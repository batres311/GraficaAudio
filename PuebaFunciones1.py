import librosa
#import librosa.display
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import pyaudio #Libreria que ayuda para obtener el audio y darle formato
import wave  #Permite leer y escribir archivos wav
#import winsound #Permite acceder a la maquinaria básica de reproducción de sonidos proporcionada por la plataformas Windows.
import scipy.io.wavfile as waves #libreria importante para los datos del audio
from datetime import datetime
from ctypes import *
from contextlib import contextmanager
import RPi.GPIO as GPIO
import yaml
import shutil



with open("variables.yaml", "r") as f:
    yaml_content = yaml.full_load(f)

WAVEFORM_path_export1 =yaml_content["WAVEFORM"]["Carpetaok"]
WAVEFORM_path_export2 = yaml_content["WAVEFORM"]["Carpetanok"]
SPECTROGRAM_path_export1=yaml_content["SPECTROGRAM"]["Carpetaok"]
SPECTROGRAM_path_export2=yaml_content["SPECTROGRAM"]["Carpetanok"]
GREYSPECTROGRAM_path_export1=yaml_content["GREYSPECTROGRAM"]["Carpetaok"]
GREYSPECTROGRAM_path_export2=yaml_content["GREYSPECTROGRAM"]["Carpetanok"]
MELSPECTROGRAM_path_export1=yaml_content["MELSPECTROGRAM"]["Carpetaok"]
MELSPECTROGRAM_path_export2=yaml_content["MELSPECTROGRAM"]["Carpetanok"]
CHROMAGRAM_path_export1=yaml_content["CHROMAGRAM"]["Carpetaok"]
CHROMAGRAM_path_export2=yaml_content["CHROMAGRAM"]["Carpetanok"]
MFCC_path_export1=yaml_content["MFCC"]["Carpetaok"]
MFCC_path_export2=yaml_content["MFCC"]["Carpetanok"]
DELTA_MFCC_path_export1=yaml_content["DELTA_MFCC"]["Carpetaok"]
DELTA_MFCC_path_export2=yaml_content["DELTA_MFCC"]["Carpetanok"]
DELTA2_MFCC_path_export1=yaml_content["DELTA2_MFCC"]["Carpetaok"]
DELTA2_MFCC_path_export2=yaml_content["DELTA2_MFCC"]["Carpetanok"]
FvsA_path_export1=yaml_content["FvsA"]["Carpetaok"]
FvsA_path_export2=yaml_content["FvsA"]["Carpetanok"]
AMPLITUDEENV_path_export1=yaml_content["AMPLITUDEENV"]["Carpetaok"]
AMPLITUDEENV_path_export2=yaml_content["AMPLITUDEENV"]["Carpetanok"]
RMSE_path_export1=yaml_content["RMSE"]["Carpetaok"]
RMSE_path_export2=yaml_content["RMSE"]["Carpetanok"]
ZCR_path_export1=yaml_content["ZCR"]["Carpetaok"]
ZCR_path_export2=yaml_content["ZCR"]["Carpetanok"]
BER_path_export1=yaml_content["BER"]["Carpetaok"]
BER_path_export2=yaml_content["BER"]["Carpetanok"]
SpecCent_path_export1=yaml_content["SpecCent"]["Carpetaok"]
SpecCent_path_export2=yaml_content["SpecCent"]["Carpetanok"]
Bandwidth_path_export1=yaml_content["Bandwidth"]["Carpetaok"]
Bandwidth_path_export2=yaml_content["Bandwidth"]["Carpetanok"]
SpecContrast_path_export1=yaml_content["SpecContrast"]["Carpetaok"]
SpecContrast_path_export2=yaml_content["SpecContrast"]["Carpetanok"]
SpecRollOff_path_export1=yaml_content["SpecRollOff"]["Carpetaok"]
SpecRollOff_path_export2=yaml_content["SpecRollOff"]["Carpetanok"]
PolyFeatures_path_export1=yaml_content["PolyFeatures"]["Carpetaok"]
PolyFeatures_path_export2=yaml_content["PolyFeatures"]["Carpetanok"]
Tonnetz_path_export1=yaml_content["Tonnetz"]["Carpetaok"]
Tonnetz_path_export2=yaml_content["Tonnetz"]["Carpetanok"]

Empezar = 11
Detener  = 13

Bosch_path_export = yaml_content["BuenoMalo"]["Bueno"]
BlackDecker_path_export = yaml_content["BuenoMalo"]["Malo"]

FRAME_SIZE = yaml_content["Frame_size"]
HOP_LENGTH = yaml_content["Hop_lenght"]
FRAME_RATE = yaml_content["Frame_rate"]
CHANNELS = yaml_content["Channels"]
NUMBER_MELS = yaml_content["Number_Mels"]
N_FTT = yaml_content["N_fft"]
N_MFCC = yaml_content["Number_MFCCs"]
HOP_SIZE= yaml_content["Hop_size"]
NOMBREGRABACION=yaml_content["NomGrabacion"]

duracion=5 #Periodo de grabacion de 5 segundos
FechaHoraAUDIO=datetime.now()
FechaHoraAUDIO=FechaHoraAUDIO.replace(microsecond=0)
FechaHoraAUDIOFormat=FechaHoraAUDIO.strftime("%Y_%m_%d_%H_%M_%S")
archivo=NOMBREGRABACION+"_"+FechaHoraAUDIOFormat +".wav"


def setup():
	GPIO.setwarnings(False) 
	GPIO.setmode(GPIO.BOARD)       # Numbers GPIOs by physical location
	   # Set Green Led Pin mode to output
	GPIO.setup(Detener, GPIO.IN, pull_up_down=GPIO.PUD_UP)      # Set Red Led Pin mode to output
	GPIO.setup(Empezar, GPIO.IN, pull_up_down=GPIO.PUD_UP) 

def LoadAudio_Turn2Decibels(clip):
    y, sr = librosa.load(clip) 
    D = librosa.stft(y) 
    # STFT of y 
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max) 
    #, ref=np.max

    return y,S_db,sr

def guardarimagen(path_export1,path_export2,res,NombreImag,fig):
    audio_filename=archivo
    image_filename_to_save = str(audio_filename).replace(".wav", "_", 1) + NombreImag+".png" 
    if not os.path.exists(path_export1): 
        os.makedirs(path_export1, exist_ok=True) 
    if not os.path.exists(path_export2): 
        os.makedirs(path_export2, exist_ok=True)
    if res=='ok'or res=='OK' or res=='Ok':
        image_filename_to_save2 ="Bosch"+image_filename_to_save 
        fig.savefig(os.path.join(path_export1,image_filename_to_save2))
    else:
        image_filename_to_save2 ="BlackDecker"+image_filename_to_save 
        fig.savefig(os.path.join(path_export2,image_filename_to_save2))
  

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


ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)

def py_error_handler(filename, line, function, err, fmt):
    pass

c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

@contextmanager
def noalsaerr():
    asound = cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(c_error_handler)
    yield
    asound.snd_lib_error_set_handler(None)

with noalsaerr():
    audio=pyaudio.PyAudio() #Iniciamos pyaudio
#Abrimos corriente o flujo

def loop(audio):
    while True: 
        if GPIO.input(Empezar)==0:                                                                                                                                                                                                                                                                                  
            stream=audio.open(format=pyaudio.paInt16,channels=2,
                                rate=44100,input=True, #rate es la frecuencia de muestreo 44.1KHz
                                frames_per_buffer=1024)
                        
            print("Grabando ...") #Mensaje de que se inicio a grabar
            frames=[] #Aqui guardamos la grabacion
            #for i in range(0,int(44100/1024*duracion)):
            while True:
                data=stream.read(1024)
                frames.append(data)

                if GPIO.input(Detener)==0: 
                    stream.stop_stream()    #Detener grabacion
                    stream.close()          #Cerramos stream
                    audio.terminate()
                    #print("La grabacion ha terminado ") #Mensaje de fin de grabación

                    waveFile=wave.open(archivo,'wb') #Creamos nuestro archivo
                    waveFile.setnchannels(2) #Se designan los canales
                    waveFile.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
                    waveFile.setframerate(44100) #Pasamos la frecuencia de muestreo
                    waveFile.writeframes(b''.join(frames))
                    waveFile.close() #Cerramos el archivo
                    break
            break
setup()
print("Listo para grabar presiona el boton ")
loop(audio)
print("La grabacion ha terminado ")
#clip=(r'/home/pi/python-projects/AudioLibrosaTest1/GraficaAudio/PruebaAudio1.wav')
#winsound.PlaySound(clip,winsound.SND_FILENAME)
#clip=('PruebaAudio1.wav')

res=input("Ingresa ok si es buena grabacion y nok si es mala: ")
y,S_db,sr=LoadAudio_Turn2Decibels(archivo)
if not os.path.exists(Bosch_path_export): 
    os.makedirs(Bosch_path_export, exist_ok=True) 
if not os.path.exists(BlackDecker_path_export): 
    os.makedirs(BlackDecker_path_export, exist_ok=True)

if res=='ok' or res=='OK' or res=='Ok':
    shutil.move(archivo, Bosch_path_export+"/"+archivo)
    os.rename(Bosch_path_export+"/"+archivo, Bosch_path_export+"/"+"Bosch"+archivo)
else:
     shutil.move(archivo, BlackDecker_path_export+"/"+archivo)
     os.rename(BlackDecker_path_export+"/"+archivo, BlackDecker_path_export+"/"+"BlackDecker"+archivo)   

""" Waveform"""
# Simple WAVEFORM to check clip trimming accuracy 
fig, ax = plt.subplots() 
img = librosa.display.waveshow(y, sr=sr) 
ax.set(title='WAVEFORM') 
#The first strips off any trailing slashes, the second gives you the last part of the path. 
guardarimagen(WAVEFORM_path_export1,WAVEFORM_path_export2,res,'waveform',fig)
plt.close()

"""Amplitude envelope"""
#FRAME_SIZE = 1024
#HOP_LENGTH = 512

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
guardarimagen(AMPLITUDEENV_path_export1,AMPLITUDEENV_path_export2,res,'AmplitudEnvelope',fig)
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
guardarimagen(RMSE_path_export1,RMSE_path_export2,res,'RMSE',fig)
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
guardarimagen(ZCR_path_export1,ZCR_path_export2,res,'ZCR',fig)
plt.close()

"""Frequency vs Amplitude"""
#Frequency vs amplitude graph
fft=np.fft.fft(y)

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
guardarimagen(FvsA_path_export1,FvsA_path_export2,res,'FvsA',fig)
plt.close()

"""Spectrogram"""
# SPECTROGRAM representation - object-oriented interface 
plt.figure(figsize=(25, 10))
fig, ax = plt.subplots() 
img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear') 
img = librosa.display.specshow(S_db, x_axis='time', y_axis='log') 
plt.colorbar(format="%+2.f")
ax.set(title='SPECTROGRAM') 
guardarimagen(SPECTROGRAM_path_export1,SPECTROGRAM_path_export2,res,'Spectrogram',fig)
plt.close()

"""Grey Spectrogram"""
# SPECTROGRAM representation - object-oriented interface 
fig, ax = plt.subplots() 
img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear') 
img = librosa.display.specshow(S_db, x_axis='time', y_axis='log', cmap='gray_r') 
plt.colorbar(format="%+2.f")
ax.set(title='GREY SPECTROGRAM') 
guardarimagen(GREYSPECTROGRAM_path_export1,GREYSPECTROGRAM_path_export2,res,'Grey Spectrogram',fig)
plt.close()

"""Mel Spectrogram"""
#Extracting Mel Spectrogram
mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FTT, hop_length=HOP_LENGTH, n_mels=NUMBER_MELS)
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

plt.figure(figsize=(25, 10))
fig, ax = plt.subplots() 
img=librosa.display.specshow(log_mel_spectrogram, 
                         x_axis="time",
                         y_axis="mel", 
                         sr=sr)
plt.colorbar(format="%+2.f")
ax.set(title='SPECTROGRAM') 
guardarimagen(MELSPECTROGRAM_path_export1,MELSPECTROGRAM_path_export2,res,'MelSpectrogram',fig)
plt.close()

"""Chromogram"""
#CHROMAGRAM representation - object-oriented interface 
CHROMAGRAM = librosa.feature.chroma_cqt(y=y, sr=sr) 
fig, ax = plt.subplots() 
img = librosa.display.specshow(CHROMAGRAM, y_axis='chroma', x_axis='time') 
plt.colorbar(format="%+2.f")
ax.set(title='CHROMAGRAM') 
guardarimagen(CHROMAGRAM_path_export1,CHROMAGRAM_path_export2,res,'Chromogram',fig)
plt.close()

"""MFCCs"""
#MFCC representation - object-oriented interface 
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=1200) 
fig, ax = plt.subplots() 
img = librosa.display.specshow(mfccs, x_axis='time') 
plt.colorbar(format="%+2.f")
ax.set(title='Mel-frequency cepstral coefficients (MFCCs)') 
guardarimagen(MFCC_path_export1,MFCC_path_export2,res,'MFCCs',fig)
plt.close()

"""Delta MFCCs"""
delta_mfccs = librosa.feature.delta(mfccs)

plt.figure(figsize=(25, 10))
fig, ax = plt.subplots() 
img = librosa.display.specshow(delta_mfccs, x_axis='time',sr=sr) 
plt.colorbar(format="%+2.f")
ax.set(title='Delta Mel-frequency cepstral coefficients (MFCCs)') 
guardarimagen(DELTA_MFCC_path_export1,DELTA_MFCC_path_export2,res,'DeltaMFCCs',fig)
plt.close()

"""Delta2 MFCCs"""
delta2_mfccs = librosa.feature.delta(mfccs, order=2)

plt.figure(figsize=(25, 10))
fig, ax = plt.subplots() 
img = librosa.display.specshow(delta2_mfccs, x_axis='time',sr=sr) 
plt.colorbar(format="%+2.f")
ax.set(title='Delta2 Mel-frequency cepstral coefficients (MFCCs)') 
guardarimagen(DELTA2_MFCC_path_export1,DELTA2_MFCC_path_export2,res,'Delta2MFCCs',fig)
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
guardarimagen(BER_path_export1,BER_path_export2,res,'Band Energy Ratio',fig)
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
guardarimagen(SpecCent_path_export1,SpecCent_path_export2,res,'Spectral Centroid',fig)
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
guardarimagen(Bandwidth_path_export1,Bandwidth_path_export2,res,'Bandwidth',fig)
plt.close()

"""Spectral Contrast"""
S = np.abs(librosa.stft(y))
contrast = librosa.feature.spectral_contrast(S=S, sr=sr)

fig, ax = plt.subplots(nrows=2, sharex=True)
img1 = librosa.display.specshow(librosa.amplitude_to_db(S,
                                                 ref=np.max),
                         y_axis='log', x_axis='time', ax=ax[0])
fig.colorbar(img1, ax=[ax[0]], format='%+2.0f dB')
ax[0].set(title='Power spectrogram')
ax[0].label_outer()
img2 = librosa.display.specshow(contrast, x_axis='time', ax=ax[1])
fig.colorbar(img2, ax=[ax[1]])
ax[1].set(ylabel='Frequency bands', title='Spectral contrast')
guardarimagen(SpecContrast_path_export1,SpecContrast_path_export2,res,'Spectral Contrast',fig)
plt.close()

""" Spectral Flatness"""

#From time-series input
flatness = librosa.feature.spectral_flatness(y=y)
flatness

#From spectrogram input
S, phase = librosa.magphase(librosa.stft(y))
librosa.feature.spectral_flatness(S=S)

#From power spectrogram input
S_power = S ** 2
librosa.feature.spectral_flatness(S=S_power, power=1.0)

"""Spectral RollOff"""
# Approximate maximum frequencies with roll_percent=0.85 (default)
librosa.feature.spectral_rolloff(y=y, sr=sr)

# Approximate maximum frequencies with roll_percent=0.99
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.99)
rolloff

# Approximate minimum frequencies with roll_percent=0.01
rolloff_min = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.01)
rolloff_min

fig, ax = plt.subplots()
librosa.display.specshow(S_db, y_axis='log', x_axis='time', ax=ax)
ax.plot(librosa.times_like(rolloff), rolloff[0], label='Roll-off frequency (0.99)')
ax.plot(librosa.times_like(rolloff), rolloff_min[0], color='w',
        label='Roll-off frequency (0.01)')
ax.legend(loc='lower right')
ax.set(title='log Power spectrogram')

guardarimagen(SpecRollOff_path_export1,SpecRollOff_path_export2,res,'Spectral Rolloff',fig)
plt.close()

"""Poly Features"""
p0 = librosa.feature.poly_features(S=S, order=0)
p1 = librosa.feature.poly_features(S=S, order=1)
p2 = librosa.feature.poly_features(S=S, order=2)

fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(8, 8))
times = librosa.times_like(p0)
ax[0].plot(times, p0[0], label='order=0', alpha=0.8)
ax[0].plot(times, p1[1], label='order=1', alpha=0.8)
ax[0].plot(times, p2[2], label='order=2', alpha=0.8)
ax[0].legend()
ax[0].label_outer()
ax[0].set(ylabel='Constant term ')
ax[1].plot(times, p1[0], label='order=1', alpha=0.8)
ax[1].plot(times, p2[1], label='order=2', alpha=0.8)
ax[1].set(ylabel='Linear term')
ax[1].label_outer()
ax[1].legend()
ax[2].plot(times, p2[0], label='order=2', alpha=0.8)
ax[2].set(ylabel='Quadratic term')
ax[2].legend()
librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                         y_axis='log', x_axis='time', ax=ax[3])
guardarimagen(PolyFeatures_path_export1,PolyFeatures_path_export2,res,'Poly features',fig)
plt.close()

"""Tonnetz"""
y = librosa.effects.harmonic(y)
tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
tonnetz

fig, ax = plt.subplots(nrows=2, sharex=True)
img1 = librosa.display.specshow(tonnetz,
                                y_axis='tonnetz', x_axis='time', ax=ax[0])
ax[0].set(title='Tonal Centroids (Tonnetz)')
ax[0].label_outer()
img2 = librosa.display.specshow(librosa.feature.chroma_cqt(y=y, sr=sr),
                                y_axis='chroma', x_axis='time', ax=ax[1])
ax[1].set(title='Chroma')
fig.colorbar(img1, ax=[ax[0]])
fig.colorbar(img2, ax=[ax[1]])

guardarimagen(Tonnetz_path_export1,Tonnetz_path_export2,res,'Tonnetz',fig)
plt.close()