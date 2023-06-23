#!/bin/bash

TRAIN_DATA=/home/kenny/avito_tech/data/raw/train.tar
VAL_DATA=/home/kenny/avito_tech/data/raw/val.csv
OUTPUT=/home/kenny/avito_tech/data/processed
STOPWORDS=/home/kenny/avito_tech/data/raw/stopwords.txt

EXECUTABLE=$1
CHUNK_SIZE=$2

python $EXECUTABLE --train-data-path $VAL_DATA --output-dir-path $OUTPUT --chunk-size $CHUNK_SIZE --path-to-stopwords $STOPWORDS
