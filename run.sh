#!/bin/bash

if [ "$1" = "DistilBERT" ]; then
    python train.py -n DistilBERT --train_record_file=./data/train.npz \
    --batch_size=8 --lr_1=2e-5 --use_img=False
elif [ "$1" = "ALBERT" ]; then
    python train.py -n ALBERT --train_record_file=./data/train_ALBERT.npz \
    --dev_record_file=./data/dev_ALBERT.npz \
    --batch_size=8 --lr_1=1e-3 --lr_2=3e-5
else
    echo "Invalid Option Selected"
fi