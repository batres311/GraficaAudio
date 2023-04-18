import librosa
import os
import numpy as np
import matplotlib.pyplot as plt

WAVEFORM_path_export = 'waveform'
SPECTROGRAM_path_export='spectogram'
CHROMAGRAM_path_export='chromagram'
MFCC_path_export='mfcc'
AMPLITUDEENV_path_export='amplitude envelope'
RMSE_path_export='root mean square energy'
ZCR_path_export='zero croosing rate'
clip = (r'C:\Users\BHC4SLP\Documents\Python Projects\Proyecto2-GraficaAudio\PruebaAudio1.wav')


y, sr = librosa.load(clip) 
D = librosa.stft(y) 
# STFT of y 
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max) 

# Simple WAVEFORM to check clip trimming accuracy 
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

#Calculatin amplitude envelope
FRAME_SIZE = 1024
HOP_LENGTH = 512

def amplitude_envelope(signal, frame_size, hop_length):
    """Calculate the amplitude envelope of a signal with a given frame size nad hop length."""
    amplitude_envelope = []
    
    # calculate amplitude envelope for each frame
    for i in range(0, len(signal), hop_length): 
        amplitude_envelope_current_frame = max(signal[i:i+frame_size]) 
        amplitude_envelope.append(amplitude_envelope_current_frame)
    
    return np.array(amplitude_envelope)   

def fancy_amplitude_envelope(signal, frame_size, hop_length):
    """Fancier Python code to calculate the amplitude envelope of a signal with a given frame size."""
    return np.array([max(signal[i:i+frame_size]) for i in range(0, len(signal), hop_length)])

# number of frames in amplitude envelope
ae_y = amplitude_envelope(y, FRAME_SIZE, HOP_LENGTH)
len(ae_y)

#Visualizing amplitud envelope
frames = range(len(ae_y))
t = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)

# amplitude envelope is graphed in red

plt.figure(figsize=(15, 17))

fig, ax = plt.subplots()
img=librosa.display.waveshow(y, alpha=0.5)
plt.plot(t, ae_y, color="r")
#plt.ylim((-1, 1))
ax.set(title="Amplitude envelope")
#The first strips off any trailing slashes, the second gives you the last part of the path. 
audio_filename = os.path.basename(os.path.normpath(clip)) 
image_filename_to_save = str(audio_filename).replace(".wav", "-", 1) + "AmplitudeEnvelope.png" 
if not os.path.exists(AMPLITUDEENV_path_export): 
    os.makedirs(AMPLITUDEENV_path_export) 
fig.savefig(os.path.join(AMPLITUDEENV_path_export,image_filename_to_save)) 
plt.close()

#Root-mean-squared energy with Librosa
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
audio_filename = os.path.basename(os.path.normpath(clip)) 
image_filename_to_save = str(audio_filename).replace(".wav", "-", 1) + "RootMeanSquareEnergy.png" 
if not os.path.exists(RMSE_path_export): 
    os.makedirs(RMSE_path_export) 
fig.savefig(os.path.join(RMSE_path_export,image_filename_to_save)) 
plt.close()

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
audio_filename = os.path.basename(os.path.normpath(clip)) 
image_filename_to_save = str(audio_filename).replace(".wav", "-", 1) + "ZeroCroosingRate.png" 
if not os.path.exists(ZCR_path_export): 
    os.makedirs(ZCR_path_export) 
fig.savefig(os.path.join(ZCR_path_export,image_filename_to_save)) 
plt.close()

# SPECTROGRAM representation - object-oriented interface 
fig, ax = plt.subplots() 
img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax) 
img = librosa.display.specshow(S_db, x_axis='time', y_axis='log', ax=ax, cmap='gray_r') 
ax.set(title='SPECTROGRAM') 
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