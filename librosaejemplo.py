import librosa
import os
import numpy as np
import matplotlib.pyplot as plt

WAVEFORM_path_export = 'waveform'
SPECTROGRAM_path_export='spectogram'
CHROMAGRAM_path_export='chromagram'
MFCC_path_export='mfcc'
clip = (r'C:\Users\BHC4SLP\Documents\Python Projects\Proyecto2-GraficaAudio\PruebaAudio1.wav')


y, sr = librosa.load(clip) 
D = librosa.stft(y) 
# STFT of y 
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max) 

# # Simple WAVEFORM to check clip trimming accuracy 
fig, ax = plt.subplots() 
img = librosa.display.waveshow(y, sr=sr, x_axis='time') 
ax.set(title='WAVEFORM') 
#The first strips off any trailing slashes, the second gives you the last part of the path. 
audio_filename = os.path.basename(os.path.normpath(clip)) 
image_filename_to_save = str(audio_filename).replace(".wav", "-", 1) + "WAVEFORM.png" 
if not os.path.exists(WAVEFORM_path_export): 
    os.makedirs(WAVEFORM_path_export) 
fig.savefig(os.path.join(WAVEFORM_path_export,image_filename_to_save)) 
plt.close()

# SPECTROGRAM representation - object-oriented interface 
fig, ax = plt.subplots() 
img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax) 
img = librosa.display.specshow(S_db, x_axis='time', y_axis='log', ax=ax, cmap='gray_r') 
ax.set(title='SPECTROGRAMgram') 
#The first strips off any trailing slashes, the second gives you the last part of the path. 
audio_filename = os.path.basename(os.path.normpath(clip)) 
image_filename_to_save = str(audio_filename).replace(".flac", "-", 1) + "SPECTROGRAM.png" 
if not os.path.exists(SPECTROGRAM_path_export): 
    os.makedirs(SPECTROGRAM_path_export) 
fig.savefig(os.path.join(SPECTROGRAM_path_export,image_filename_to_save)) 
plt.close()

#CHROMAGRAM representation - object-oriented interface 
CHROMAGRAM = librosa.feature.chroma_cqt(y=y, sr=sr) 
fig, ax = plt.subplots() 
img = librosa.display.specshow(CHROMAGRAM, y_axis='chroma', x_axis='time', ax=ax) 
ax.set(title='CHROMAGRAM') 
audio_filename = os.path.basename(os.path.normpath(clip)) 
image_filename_to_save = str(audio_filename).replace(".flac", "-", 1) + "CHROMAGRAM.png" 
if not os.path.exists(CHROMAGRAM_path_export): 
    os.makedirs(CHROMAGRAM_path_export) 
fig.savefig(os.path.join(CHROMAGRAM_path_export,image_filename_to_save)) 
plt.close()

#MFCC representation - object-oriented interface 
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=1200) 
fig, ax = plt.subplots() 
img = librosa.display.specshow(mfccs, x_axis='time') 
ax.set(title='Mel-frequency cepstral coefficients (MFCCs)') 
audio_filename = os.path.basename(os.path.normpath(clip)) 
image_filename_to_save = str(audio_filename).replace(".flac", "-", 1) + "MFCC.png" 
if not os.path.exists(MFCC_path_export): 
    os.makedirs(MFCC_path_export) 
fig.savefig(os.path.join(MFCC_path_export,image_filename_to_save)) 
plt.close()