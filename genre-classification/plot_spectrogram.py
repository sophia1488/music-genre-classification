from matplotlib import pyplot as plt
import librosa
import numpy as np
import torchaudio
import torch


def extract_spectrogram(sr, clip):
	num_channels = 3
	window_sizes = [25, 50, 100]
	hop_sizes = [10, 25, 50]

	specs = []
	for i in range(num_channels):
		window_length = int(round(window_sizes[i]*sr/1000))
		hop_length = int(round(hop_sizes[i]*sr/1000))

		clip = torch.Tensor(clip)
		spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=2205, win_length=window_length, hop_length=hop_length, n_mels=128)(clip) #Check this otherwise use 2400
		eps = 1e-6
		spec = spec.numpy()
		spec = np.log(spec+ eps)
		# Resize is scaling, not crop
		spec = np.asarray(torchvision.transforms.Resize((128, 1500))(Image.fromarray(spec)))
		return spec
    
    
sampling_rate = 22050
f, axs = plt.subplots(5,2,figsize=(15,15))

MAP = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

for i, genre in enumerate(MAP):
    audio = f'../Data/genres_original/{genre}/{genre}.00000.wav'
    clip, sr = librosa.load(audio, sr=sampling_rate)
    spec = extract_spectrogram(sampling_rate, clip)
    spec = spec[:, 500:800]   # randomly crop it
    
    r, c = i // 2, i % 2
    plt.subplot2grid((5,2), (r,c))  
    plt.title(genre)  
    
    if r == 4:
        plt.xlabel('Time (resized)')
    plt.ylabel('Log Mel Frequency')
    plt.tight_layout()
    plt.imshow(spec, interpolation='nearest', aspect='auto')
