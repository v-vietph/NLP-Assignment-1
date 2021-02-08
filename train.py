import torch
import numpy as np
import pandas as pd 
from torch.utils.data import Dataset, DataLoader
from data_loader import WordEmbeddingDataset


from RNN import BasicRNN, train_RNN
from LSTM import BasicLSTM, train_LSTM
from GRU import BasicGRU, train_GRU

import argparse



use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.autograd.set_detect_anomaly(True)

def train(train_data_dir = './data/train.txt',glove_dimension=50, model_to_train = 'RNN', n_epochs=20, learning_rate=0.001, hidden_size=50):

    word_dataset = WordEmbeddingDataset(train_data_dir, dimension=glove_dimension)
    dataloader = DataLoader(word_dataset, batch_size=1,
                            shuffle=True, num_workers=4)
    print("Training model", model_to_train)
    if model_to_train == 'RNN':
        train_RNN(dataloader=dataloader, device=device,n_epochs = n_epochs, learning_rate = learning_rate, hidden_size=hidden_size, input_size=glove_dimension)
    elif model_to_train == 'LSTM':
        train_LSTM(dataloader=dataloader, device=device,n_epochs = n_epochs, learning_rate = learning_rate, hidden_size=hidden_size, input_size=glove_dimension)
    elif model_to_train == 'GRU':
        train_GRU(dataloader=dataloader, device=device,n_epochs = n_epochs, learning_rate = learning_rate, hidden_size=hidden_size, input_size=glove_dimension)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train_data',default="./data/train.txt", help='Test data dir')
    parser.add_argument('--model_to_train', default="RNN", help='Which model to test: RNN, GRU or LSTM')
    parser.add_argument('--n_epochs',type = int, default=20, help='Number of epochs')
    parser.add_argument('--learning_rate',type = float, default=0.001, help='Number of epochs')
    parser.add_argument('--hidden_size',type = int, default=50, help='Number of epochs')
    parser.add_argument('--pretrained_size',type = int, default=50, help='Dimension of Glove pretrain')
    args = parser.parse_args()
    train(args.train_data, args.pretrained_size, args.model_to_train, args.n_epochs, args.learning_rate, args.hidden_size)

