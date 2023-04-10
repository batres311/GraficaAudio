# Importar librerias
import wave
import matplotlib.pyplot as plt
import numpy as np

# Cargar archivo de audio
audio=wave.open(r'C:\Users\BHC4SLP\Documents\Proyecto Audio\mi_explosion_03_hpx.wav',"rb")

# Obtener variables
sample_freq=audio.getframerate()
n_samples=audio.getnframes()
signal_wave=audio.readframes(-1)

audio.close() #Cerramos el archivo de audio

# Duracion del audio
audio_duration=n_samples/sample_freq

# Signal array
signal_array=np.frombuffer(signal_wave,dtype=np.int16)
times=np.linspace(0,audio_duration,num=n_samples)

# Graficar señal de audio

plt.figure(figsize=(15,5))
plt.plot(times,signal_array)
plt.title("AUDIO SIGNAL")
plt.ylabel("Signal wave")
plt.xlabel("Time in seconds")
plt.xlim(0,audio_duration)
plt.show()