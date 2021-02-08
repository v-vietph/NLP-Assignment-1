import torch
import numpy as np
from torch import nn
import math

class BasicLSTM(nn.Module):
    def __init__(self, input_size=50, hidden_size=50, output_size = 50, device='cpu'):
        super(BasicLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        
        #i_t
        self.U_i = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))
        
        #f_t
        self.U_f = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))
        
        #c_t
        self.U_c = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(hidden_size))
        
        #o_t
        self.U_o = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))
        
        self.out_linear = nn.Linear(hidden_size, 2)
        self.softmax = nn.Softmax(dim=1)

        self.init_weights()

        self.hidden_state = torch.rand(1, hidden_size, device=device) # 1 X n_neurons
        self.cell_state = torch.rand(1, hidden_size, device=device) # 1 X n_neurons
    
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input_vector):
        
        f = torch.sigmoid(torch.mm(input_vector, self.U_f) + torch.mm(self.hidden_state, self.V_f) + self.b_f)
        i = torch.sigmoid(torch.mm(input_vector, self.U_i) + torch.mm(self.hidden_state, self.V_i) + self.b_i)
        o = torch.sigmoid(torch.mm(input_vector, self.U_o) + torch.mm(self.hidden_state, self.V_o) + self.b_o)

        c_hat = torch.tanh(torch.mm(input_vector, self.U_c) + torch.mm(self.hidden_state, self.V_c) + self.b_c)

        self.cell_state = f * self.cell_state + i * c_hat
        self.hidden_state = o * torch.tanh(self.cell_state)

        output = self.out_linear(self.hidden_state)
        pre_label = self.softmax(output)

        return pre_label

def train_LSTM(dataloader, device, input_size=50, hidden_size=50, n_epochs = 20, learning_rate = 0.001):
    lstm = BasicLSTM(input_size=input_size, hidden_size=hidden_size, device=device)
    print("input_size", input_size)
    print("hidden_size", hidden_size)
    print("n_epochs", n_epochs)
    print("device", device)
    print("learning_rate", learning_rate)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lstm.parameters())
    
    for epoch in range(1, n_epochs + 1):
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        for embedding_tensors, label_tensor in dataloader:
            if label_tensor == -1:
                continue
            optimizer.zero_grad() # Clears existing gradients from previous epoch
            for tensor in embedding_tensors[0]:
                tensor = torch.reshape(tensor, (1,-1))
                output = lstm.forward(tensor.to(device))
            # print(output, label_tensor)
            loss = criterion(output, label_tensor)
            loss.backward(retain_graph=True) # Does backpropagation and calculates gradients
            # print(loss.item())
            # optimizer.step()
            for p in lstm.parameters():
                p.data.add_(p.grad.data, alpha=-learning_rate)
            # print("Loss: {:.4f}".format(loss.item()))

        print("Loss: {:.4f}".format(loss.item()))
        PATH = './model/LSTM-'+str(hidden_size)+"-"+str(input_size)+"-"+str(epoch)+'.pt'
        torch.save(lstm, PATH) 
    # lstm.save_state_dict(PATH)

def test_LSTM(dataloader, model, device, hidden_size=50):
    correct = 0
    count = 0
    for embedding_tensors, label_tensor in dataloader:
        hidden_state = torch.zeros(1, hidden_size, device=device) # 1 X n_neurons
        cell_state = torch.zeros(1, hidden_size, device=device) # 1 X n_neurons
        if label_tensor == -1:
            continue
        for tensor in embedding_tensors[0]:
            tensor = torch.reshape(tensor, (1,-1))
            output = model.forward(tensor.to(device))
        if(float(torch.argmax(output) -  label_tensor) == 0):
            correct+=1
        count+=1
    print("ACC", correct/count)