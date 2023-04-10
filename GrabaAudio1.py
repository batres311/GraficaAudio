#pip install pyaudio  ==> para instalar pyaudio
#Autor : Marco Jhoel Churata Torres.
#Titulo:Grabacion_de_audio_con_Python

import pyaudio
import wave

duracion=5
archivo="ejemplo2.wav"

audio=pyaudio.PyAudio()

stream=audio.open(format=pyaudio.paInt16,channels=2,
					rate=44100,input=True,
					frames_per_buffer=1024)
					
print("Grabando ...")
frames=[]

for i in range(0,int(44100/1024*duracion)):
	data=stream.read(1024)
	frames.append(data)
	
print("Grabacion a terminado ")

stream.stop_stream()
stream.close()
audio.terminate()

waveFile=wave.open(archivo,'wb')
waveFile.setnchannels(2)
waveFile.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
waveFile.setframerate(44100)
waveFile.writeframes(b''.join(frames))
waveFile.close()