import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
import pyaudio #Libreria que ayuda para obtener el audio y darle formato
import wave  #Permite leer y escribir archivos wav
#import winsound #Permite acceder a la maquinaria básica de reproducción de sonidos proporcionada por la plataformas Windows.
import scipy.io.wavfile as waves #libreria importante para los datos del audio
import scipy.fftpack as fourier #libreria para pasar al dominio de la frecuencia de forma sencilla
from ctypes import *
from contextlib import contextmanager
from datetime import datetime

MFCC_path_export1='mfcc/OK'
MFCC_path_export2='mfcc/NOK'

FRAME_SIZE = 1024
HOP_LENGTH = 512

duracion=4 #Periodo de grabacion de 5 segundos
archivo="PruebaAudio1.wav" #Se define el nombre del archivo donde se guardara la grabación

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
stream=audio.open(format=pyaudio.paInt16,channels=2,
					rate=44100,input=True, #rate es la frecuencia de muestreo 44.1KHz
					frames_per_buffer=1024)
					
print("Grabando ...") #Mensaje de que se inicio a grabar
frames=[] #Aqui guardamos la grabacion
for i in range(0,int(44100/1024*duracion)):
	data=stream.read(1024)
	frames.append(data)
	
print("La grabacion ha terminado ") #Mensaje de fin de grabación
stream.stop_stream()    #Detener grabacion
stream.close()          #Cerramos stream
audio.terminate()

waveFile=wave.open(archivo,'wb') #Creamos nuestro archivo
waveFile.setnchannels(2) #Se designan los canales
waveFile.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
waveFile.setframerate(44100) #Pasamos la frecuencia de muestreo
waveFile.writeframes(b''.join(frames))
waveFile.close() #Cerramos el archivo

clip = ('PruebaAudio1.wav')

def LoadAudio_Turn2Decibels(clip):
    y, sr = librosa.load(clip) 
    D = librosa.stft(y) 
    # STFT of y 
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max) 
    #, ref=np.max

    return y,S_db,sr

def guardarimagen(path_export1,path_export2,res,NombreImag,fig):
    audio_filename = os.path.basename(os.path.normpath(clip)) 
    FechaHora=datetime.now()
    FechaHora=FechaHora.replace(microsecond=0)
    image_filename_to_save = str(audio_filename).replace(".wav", "-", 1) + NombreImag+" "+str(FechaHora).replace(":", "-", 2) +".png" 
    if not os.path.exists(path_export1): 
        os.makedirs(path_export1, exist_ok=True) 
    if not os.path.exists(path_export2): 
        os.makedirs(path_export2, exist_ok=True)
    if res=='ok':
        fig.savefig(os.path.join(path_export1,image_filename_to_save))
    else:
        fig.savefig(os.path.join(path_export2,image_filename_to_save))



res=input("Ingresa ok si es buena grabacion y nok si es mala: ")
y,S_db,sr=LoadAudio_Turn2Decibels(clip)

"""MFCCs"""
#MFCC representation - object-oriented interface 
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=1200) 
fig, ax = plt.subplots() 
img = librosa.display.specshow(mfccs, x_axis='time') 
plt.colorbar(format="%+2.f")
ax.set(title='Mel-frequency cepstral coefficients (MFCCs)') 
guardarimagen(MFCC_path_export1,MFCC_path_export2,res,'MFCCs',fig)
plt.close()

# Simple WAVEFORM to check clip trimming accuracy 
"""fig, ax = plt.subplots() 
img = librosa.display.waveshow(y, sr=sr) 
ax.set(title='WAVEFORM') """
#The first strips off any trailing slashes, the second gives you the last part of the path. 


"""audio_filename = os.path.basename(os.path.normpath(clip)) 
image_filename_to_save = str(audio_filename).replace(".wav", "-", 1) + "WAVEFORM.png" 
if not os.path.exists(WAVEFORM_path_export): 
    os.makedirs(WAVEFORM_path_export) 
fig.savefig(os.path.join(WAVEFORM_path_export,image_filename_to_save)) 
plt.close()"""

