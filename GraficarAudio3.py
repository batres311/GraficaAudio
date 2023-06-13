import pyaudio #Libreria que ayuda para obtener el audio y darle formato
import wave  #Permite leer y escribir archivos wav
#import winsound #Permite acceder a la maquinaria básica de reproducción de sonidos proporcionada por la plataformas Windows.
import scipy.io.wavfile as waves #libreria importante para los datos del audio
import numpy as np #libreria importante para arrays y matrices
import matplotlib.pyplot as plt #libreria para graficar resultados
import scipy.fftpack as fourier #libreria para pasar al dominio de la frecuencia de forma sencilla

duracion=5 #Periodo de grabacion de 5 segundos
archivo="PruebaAudio1.wav" #Se define el nombre del archivo donde se guardara la grabación

from ctypes import *
from contextlib import contextmanager
import pyaudio

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


#audio=pyaudio.PyAudio() #Iniciamos pyaudio

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

# Acceso a archivo de audio WAV con ruta de este y reproduccion de el
filename=(r'/home/pi/python-projects/AudioLibrosaTest1/GraficaAudio/PruebaAudio1.wav')
#winsound.PlaySound(filename,winsound.SND_FILENAME)

#Empieza arreglo de audio asi podemos guardar estos datos de la grabacion dentro de el
Fs, data=waves.read(filename)
Audio_m=data[:,0] #Renombramos este arreglo como Audio_m

#

L=len(Audio_m)  #Se calcula longitud del arreglo que contiene los valores del audio
Ts=0.001        #Se declara tiempo de muestreo de 0.001 segundos
n=Ts*np.arange(0,L) #Creamos un arreglo de longitud L separado exactamente por Ts

#Procedimiento para graficar en dominio del tiempo el audio
fig, ax=plt.subplots()
plt.plot(n,Audio_m)
plt.xlabel('Tiempo (s)')
plt.ylabel('Audio')


#import scipy.fftpack as fourier
gk=fourier.fft(Audio_m) #Transformada de fourier sobre el vector con los valores del audio
M_gk=abs(gk)            #Calculo de su valor absoluto de los nuevos valores tras la transformada
M_gk=M_gk[0:L//2]       #Funcion par asi que basta con analizar la mitad de los valores
F=(Fs/L)*np.arange(0,L//2) #Se declara un arreglo hasta L medios

#Procedimiento para graficar en dominio de la frecuencia el audio y ver su espectro
fig, bx=plt.subplots()
plt.plot(F,M_gk)
plt.xlabel('Frecuencia (Hz)', fontsize='14')
plt.ylabel('Amplitud FFT', fontsize='14')
plt.show()

