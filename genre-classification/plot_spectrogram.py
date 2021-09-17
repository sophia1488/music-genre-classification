from matplotlib import pyplot as plt

sampling_rate = 22050
f, axs = plt.subplots(5,2,figsize=(15,15))

MAP = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

for i, genre in enumerate(MAP):
    audio = f'../Data/genres_original/{genre}/{genre}.00000.wav'
    clip, sr = librosa.load(audio, sr=sampling_rate)
    spec = extract_spectrogram(sampling_rate, [], clip, -1)
    spec = spec[:, 500:800]   # randomly crop it
    
    r, c = i // 2, i % 2
    plt.subplot2grid((5,2), (r,c))  
    plt.title(genre)  
    
    if r == 4:
        plt.xlabel('Time (resized)')
    plt.ylabel('Log Mel Frequency')
    plt.tight_layout()
    plt.imshow(spec, interpolation='nearest', aspect='auto')
