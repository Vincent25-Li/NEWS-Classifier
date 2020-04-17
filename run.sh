#!/bin/bash

if [ "$1" = "DistilBERT" ]; then
    python train.py -n DistilBERT --train_record_file=./data/train.npz \
    --batch_size=8
else
    echo "Invalid Option Selected"
fi