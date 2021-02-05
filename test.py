import torch
import numpy as np
import pandas as pd 
from torch.utils.data import Dataset, DataLoader
from data_loader import WordEmbeddingDataset


from RNN import BasicRNN, test_RNN
from LSTM import BasicLSTM, test_LSTM
from GRU import BasicGRU, test_GRU

import argparse

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.autograd.set_detect_anomaly(True)

def test(test_data_dir = './data/test.txt', model_to_test = 'RNN'):

    word_dataset = WordEmbeddingDataset(test_data_dir)
    dataloader = DataLoader(word_dataset, batch_size=1,
                            shuffle=True, num_workers=4)
    print("Testing model", model_to_test)
    if model_to_test == 'RNN':
        PATH = './model/model_best/RNN_model.pt'
        rnn = torch.load(PATH)
        test_RNN(dataloader, rnn, device)
    elif model_to_test == 'LSTM':
        PATH = './model/model_best/LSTM_model.pt'
        lstm = torch.load(PATH)
        test_LSTM(dataloader, lstm, device)
    elif model_to_test == 'GRU':
        PATH = './model/model_best/GRU_model.pt'
        gru = torch.load(PATH)
        test_GRU(dataloader, gru, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--test_data',default="./data/test.txt", help='Test data dir')
    parser.add_argument('--model_to_test', default="RNN", help='Which model to test: RNN, GRU or LSTM')
    args = parser.parse_args()
    test(args.test_data, args.model_to_test)
