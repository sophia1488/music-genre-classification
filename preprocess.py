import os
import argparse
import numpy as np
import torch
import torchaudio
import torchvision
from PIL import Image
import glob 
import random
import pickle as pkl
from joblib import dump
import librosa
import pandas as pd
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="Data/genres_original", type=str)
    parser.add_argument("--output_dir", default="Data", type=str)
    parser.add_argument("--sr", default=22050, type=int)

    args = parser.parse_args()
    return args
  
  
def extract_spectrogram(sr, values, clip, target):
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
        specs.append(spec)

    new_entry = {}
    new_entry["audio"] = clip.numpy()
    new_entry["values"] = np.array(specs)
    new_entry["target"] = target
    values.append(new_entry)
    
    
def extract_features(audios, sr, MAP):
    values = []
    for audio in tqdm(audios):
        try:
            clip, sr = librosa.load(audio, sr=sr)
        except:
            continue
        genre = audio.split('/')[-1].split('.')[0]
        class_idx = MAP[genre]
        extract_spectrogram(sr, values, clip, class_idx)
        #print("Finished audio {}".format(audio))
    return values
  
  
def random_split(input_dir, output_dir, class_names):
    training_audios = []
    validation_audios = []
    test_audios = []

    for _class in class_names:
        class_dir = os.path.join(input_dir, _class)
        wavs = glob.glob(f"{input_dir}/{_class}/*.wav")
        random.shuffle(wavs)
        # 8: 1: 1
        total = len(wavs)
        training_audios += wavs[:int(total*0.8)]
        validation_audios += wavs[int(total*0.8) : int(total*0.9)]
        test_audios += wavs[int(total*0.9):]
    
    # write to csv file
    df = pd.DataFrame(training_audios, columns=["path"])
    df.to_csv(f'{output_dir}/train.csv', index=False)

    df = pd.DataFrame(validation_audios, columns=["path"])
    df.to_csv(f'{output_dir}/valid.csv', index=False)

    df = pd.DataFrame(test_audios, columns=["path"])
    df.to_csv(f'{output_dir}/test.csv', index=False)

    return training_audios, validation_audios, test_audios
  

def main():
    args = get_args()
    class_names = os.listdir(args.input_dir)

    MAP = {}
    for i in range(len(class_names)):
        MAP[class_names[i]] = i
    print(MAP)
    # {'disco': 0, 'pop': 1, 'rock': 2, 'jazz': 3, 'blues': 4, 'hiphop': 5, 'metal': 6, 'classical': 7, 'country': 8, 'reggae': 9}

    if os.path.exists(f"{args.output_dir}/train.csv") and os.path.exists(f"{args.output_dir}/valid.csv") and\
        os.path.exists(f"{args.output_dir}/test.csv"):
        df = pd.read_csv(f"{args.output_dir}/train.csv")
        training_audios = df['path'].to_list()
        df = pd.read_csv(f"{args.output_dir}/valid.csv")
        validation_audios = df['path'].to_list()
        df = pd.read_csv("test.csv")
        test_audios = df['path'].to_list()
    else:
        training_audios, validation_audios, test_audios = random_split(args.input_dir, args.output_dir, class_names)
    
    # extract feature
    training_values = extract_features(training_audios, args.sr, MAP)
    with open(f"{args.output_dir}/train_128mel.pkl","wb") as handler:
        pkl.dump(training_values, handler, protocol=pkl.HIGHEST_PROTOCOL)


    validation_values = extract_features(validation_audios, args.sr, MAP)
    with open(f"{args.output_dir}/valid_128mel.pkl", "wb") as handler:
        pkl.dump(validation_values, handler, protocol=pkl.HIGHEST_PROTOCOL)

    test_values = extract_features(test_audios, args.sr, MAP)
    with open(f"{args.output_dir}/test_128mel.pkl","wb") as handler:
        pkl.dump(test_values, handler, protocol=pkl.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
