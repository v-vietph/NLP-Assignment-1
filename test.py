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

def test(test_data_dir = './data/test.txt', model_to_test = 'RNN', hidden_size=50, glove_dimension=50):

    word_dataset = WordEmbeddingDataset(test_data_dir, dimension=glove_dimension)
    dataloader = DataLoader(word_dataset, batch_size=1,
                            shuffle=True, num_workers=4)
    print("Testing model", model_to_test)
    PATH = './model/model_best/' +model_to_test+"_model_"+str(hidden_size)+"_"+str(glove_dimension)+".pt"
    if model_to_test == 'RNN':
        rnn = torch.load(PATH)
        test_RNN(dataloader, rnn, device, hidden_size=hidden_size)
    elif model_to_test == 'LSTM':
        lstm = torch.load(PATH)
        test_LSTM(dataloader, lstm, device, hidden_size=hidden_size)
    elif model_to_test == 'GRU':
        gru = torch.load(PATH)
        test_GRU(dataloader, gru, device, hidden_size=hidden_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--test_data',default="./data/test.txt", help='Test data dir')
    parser.add_argument('--model_to_test', default="RNN", help='Which model to test: RNN, GRU or LSTM')
    parser.add_argument('--hidden_size',type = int, default=50, help='Number of epochs')
    parser.add_argument('--pretrained_size',type = int, default=50, help='Dimension of Glove pretrain')
    args = parser.parse_args()
    test(args.test_data, args.model_to_test, args.hidden_size, args.pretrained_size)
