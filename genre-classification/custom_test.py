import librosa
import sys
import pickle as pkl
from preprocess import extract_spectrogram


def main():
    if len(sys.argv) != 2:
        print("Usage: python custom_dataset.py [audio file (.wav)]")
        print("{'disco': 0, 'pop': 1, 'rock': 2, 'jazz': 3, 'blues': 4, 'hiphop': 5, 'metal': 6, 'classical': 7, 'country': 8, 'reggae': 9}")
        exit(1)
        
    audio = sys.argv[1]
    
    clip, sr = librosa.load(audio)
    values = extract_spectrogram(sampling_rate, v, clip, 10)     # randomly set class_idx

    with open(f"{audio.split('.')[0]}.pkl","wb") as handler:
        pkl.dump(values, handler, protocol=pkl.HIGHEST_PROTOCOL)
