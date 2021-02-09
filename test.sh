#!/bin/bash
python test.py --model_to_test RNN --hidden_size 50 --pretrained_size 50
python test.py --model_to_test RNN --hidden_size 100 --pretrained_size 100
python test.py --model_to_test RNN --hidden_size 150 --pretrained_size 50
python test.py --model_to_test GRU --hidden_size 50 --pretrained_size 50
python test.py --model_to_test GRU --hidden_size 100 --pretrained_size 100
python test.py --model_to_test GRU --hidden_size 150 --pretrained_size 50
python test.py --model_to_test LSTM --hidden_size 50 --pretrained_size 50
python test.py --model_to_test LSTM --hidden_size 100 --pretrained_size 100
python test.py --model_to_test LSTM --hidden_size 150 --pretrained_size 50
