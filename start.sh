#!/bin/bash
pip install -r requirements.txt
python3 train.py --dataset mnist --batch_size 50 --dim 32 --o 784
