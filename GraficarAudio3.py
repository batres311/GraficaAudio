import pyaudio
import wave  #Permite leer y escribir archivos wav
import winsound #Permite acceder a la maquinaria básica de reproducción de sonidos proporcionada por la plataformas Windows.
import scipy.io.wavfile as waves
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fourier

duracion=5 #Periodo de grabacion de 5 segundos
archivo="PruebaAudio1.wav" #Se define el nombre del archivo donde se guardara la grabación

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

# Acceso a archivo de audio WAV con ruta de este y reproduccion de el
filename=(r'C:\Users\BHC4SLP\Documents\Python Projects\Proyecto2-GraficaAudio\PruebaAudio1.wav')
winsound.PlaySound(filename,winsound.SND_FILENAME)

#Empieza arreglo de audio asi podemos guardar estos datos de la grabacion dentro de el
Fs, data=waves.read(filename)
Audio_m=data[:,0] #Renombramos este arreglo como Audio_m

#

L=len(Audio_m)  #Se calcula longitud del arreglo que contiene los valores del audio
Ts=0.001        #Se declara tiempo de muestreo de 0.001 segundos
n=Ts*np.arange(0,L) #Creamos un arreglo de longitud L separado exactamente por Ts

#import matplotlib.pyplot as plt

fig, ax=plt.subplots()
plt.plot(n,Audio_m)
plt.xlabel('Tiempo (s)')
plt.ylabel('Audio')


#import scipy.fftpack as fourier
gk=fourier.fft(Audio_m)
M_gk=abs(gk)
M_gk=M_gk[0:L//2]

F=(Fs/L)*np.arange(0,L//2)
fig, bx=plt.subplots()
plt.plot(F,M_gk)
plt.xlabel('Frecuencia (Hz)', fontsize='14')
plt.ylabel('Amplitud FFT', fontsize='14')
plt.show()