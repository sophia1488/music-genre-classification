# music-genre-classification

This repo is modified from https://github.com/kamalesh0406/Audio-Classification.

I use GTZAN dataset & pre-trained DensetNet to do music genre classification.

1. Download GTZAN from [here](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification), and unzip it
2. Move *.csv to Data/

Note that one song in GTZAN dataset contains data in unknown format, make sure to delete it.
```
import os
os.remove('Data/genres_original/jazz/jazz.00054.wav')
```
