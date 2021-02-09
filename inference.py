import torch
import numpy as np
import pandas as pd 
from torch.utils.data import Dataset, DataLoader
from data_loader import WordEmbeddingDataset


from RNN import BasicRNN, test_RNN
from LSTM import BasicLSTM, test_LSTM
from GRU import BasicGRU, test_GRU

import os

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.autograd.set_detect_anomaly(True)


def word2tensor(word_list, vocab_dict, glove_dimension=50):
    embeddings = []
    
    for word in word_list:
        try:
            embeddings.append(vocab_dict[word].astype(np.float32))
        except KeyError:
            embeddings.append(np.random.rand(glove_dimension,).astype(np.float32))
    if len(embeddings) == 0:
        embeddings.append(np.random.rand(glove_dimension,).astype(np.float32))
    tensors = torch.tensor(embeddings)
    embeddings = None
    return tensors


def inference(model_to_test = 'RNN', hidden_size=50, glove_dimension=50):
    PATH = './model/model_best/' +model_to_test+"_model_"+str(hidden_size)+"_"+str(glove_dimension)+".pt"
    if not os.path.exists(PATH):
        raise ValueError(model_to_test+"_model_"+str(hidden_size)+"_"+str(glove_dimension)+".pt"+" not found")
    model = torch.load(PATH,map_location=device)
    print("Model", model_to_test+"_model_"+str(hidden_size)+"_"+str(glove_dimension)+".pt", "loaded")
    glove_path = './glove/glove.6B.'+str(glove_dimension)+'d.txt'
    frame = pd.read_csv(glove_path, sep=" ", quoting=3, header=None, index_col=0)
    vocab = {key: val.values for key, val in frame.T.items()}
    print("Glove loaded")
    
    while(True):
        in_seq = input("Input sequence or ctrl+d to finish: ")
        word_list = in_seq.split()
        tensors = word2tensor(word_list, vocab, glove_dimension)
        output = []
        if model_to_test == 'RNN' or model_to_test == 'GRU':
            hidden_state = torch.zeros(1, hidden_size, device=device)
            for tensor in tensors:
                tensor = torch.reshape(tensor, (1,-1))
                hidden_state, output = model.forward(tensor.to(device), hidden_state)
        elif model_to_test == 'LSTM':
            hidden_state = torch.rand(1, hidden_size, device=device) # 1 X n_neurons
            cell_state = torch.rand(1, hidden_size, device=device)
            for tensor in tensors:
                tensor = torch.reshape(tensor, (1,-1))
                output, hidden_state, cell_state = model.forward(tensor.to(device), hidden_state, cell_state)
        if int(torch.argmax(output)) == 0:
                print("Negative")
        else: 
            print("Positive")

if __name__ == '__main__':
    model_to_test = input("Choose your model <RNN|LSTM|GRU>: ")
    hidden_size = int(input("Choose the hidden_size <50|100|150>: "))
    pretrained_size = int(input("Choose the glove_size <50|100>:"))

    inference(model_to_test, hidden_size, pretrained_size)
