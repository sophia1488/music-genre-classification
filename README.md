# Music-Genre-Classification

This repo is modified from https://github.com/kamalesh0406/Audio-Classification.

## Download dataset
I use GTZAN dataset & pre-trained DensetNet to do music genre classification.

Download GTZAN from [here](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification), and unzip it

* Note that one song in GTZAN dataset contains data in unknown format, make sure to delete it.
```
import os
os.remove('Data/genres_original/jazz/jazz.00054.wav')
```

* I randomly split audios to train, valid, test set (8:1:1), and the paths are stored in ```Data/*.csv```

## Preprocess
```
cd genre-classification
python preprocess.py --input_dir={path to genres_original/*/*.wav} --output_dir={path to store features and csv files} --sr={sampling rate, default=22050} 
```

The command will look into ```[output_dir]``` and check if ```train.csv```, ```valid.csv```, ```test.csv``` exist.

If yes, extract features and save to ```[output_dir]/train_128mel.pkl```

(You may use my ```Data/*.csv```, or split yours in a random manner.)

Otherwise, it will split audios into train, valid, test set and store the paths in ```[output_dir]/*.csv```.

![下載](https://user-images.githubusercontent.com/47291963/133715428-b72fbe63-9278-4ed7-b833-e9832ddf14f7.png)

## Train
```
cd genre-classification
python3 main.py --name={output_directory}
```
Note that you should try setting batch size to a larger number, like 16 or 32.  (I set 8 due to the GPU limit.)

You can see log file at ```result/[name]/log.log```, the model checkpoints can also be found in that directory.

## Evaluation
```
cd genre-classification
python3 eval.py --save_dir={the directory contains checkpoints}
```
This will evaluate the model on testset.  

A confusion matrix will be saved at ```save_dir```.

<img src="https://user-images.githubusercontent.com/47291963/133106920-e12bfe15-f2c4-490b-a5ba-59b43baf1639.jpg" width="400">

I've been asked an interesting question: "what if the input spectrogram is flipped"?
```
# ./genre_classification/preprocess.py, line 46, add this to flip the tensor in time-domain
spec = np.flip(spec, 1)
```

So, here's the result.  The test accuracy drops from 1.0 to 0.8.  The performance drop is significant in terms of rock, blues, hip-hop songs.

<img src="https://user-images.githubusercontent.com/47291963/133642080-248e1417-403b-45f3-a1b7-d80b1666ab31.jpg" width="400">


## Use custom data (any .wav files) to evaluate the model
I have provided my checkpoint [here](https://drive.google.com/file/d/1C2P0y3qukEWHSPW73j9ARbT59HxYjQvb/view?usp=sharing), feel free to use it!
```
cd genre-classification
# --save_dir: only the directory, e.g. result/0913, do not provide with your checkpoint path
python3 custom_test.py --save_dir={the directory contains checkpoints} --audio={your audio file}
```

### My observations with some random Youtube songs:
For classical music, Chopin Waltz and Greig - Morning from Peer Grynt, the model can successfully classify both of them as "classical" genre.  As a classical music lover, I also find it easier to separate classical music from the rest of them lol.

For jazz, I've tried 2 songs, one with vocal (What a wonderful world), another just pure instrument.  The model cannot classify the vocal one correctly, yet it does classify the pure instrument song as jazz.  My guess is that jazz songs in GTZAN dataset are mostly (I didn't listen to all of them) instrumental, so the model may be confused when there's a human singing.

I've also tried 2 songs for reggae and blues genre respectively, the model cannot predict correctly.

So, even though the model perform decently well on GTZAN valid & test set, the model may not correctly classify all of them since the dataset is not so diverse.
