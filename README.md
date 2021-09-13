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

## Train
```
cd genre-classification
python3 train.py --name={output_directory}
```
You can see log file at ```result/[name]/log.log```, the model checkpoints can also be found in that directory.

## Evaluation
```
cd genre-classification
python3 eval.py --save_dir={the directory contains checkpoints}
```

## Use custom data (3 sec clip) to evaluate the model
TODO
