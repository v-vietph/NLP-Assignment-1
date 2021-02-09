import torch
import numpy as np
from torch import nn
import math

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class BasicGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size = 50, device='cpu'):
        super(BasicGRU, self).__init__()
        
        # self.device = device

        self.input_size = input_size
        self.hidden_size = hidden_size
        
    
        self.W_z = nn.Parameter(torch.zeros([input_size + hidden_size, hidden_size], device=device))
        self.b_z = nn.Parameter(torch.zeros([hidden_size], device=device))
        
        self.W_r = nn.Parameter(torch.zeros([input_size + hidden_size, hidden_size], device=device))
        self.b_r = nn.Parameter(torch.zeros([hidden_size], device=device))

        self.W_h = nn.Parameter(torch.zeros([input_size + hidden_size, hidden_size], device=device))
        self.b_h = nn.Parameter(torch.zeros([hidden_size], device=device))

        self.out_linear = nn.Linear(hidden_size, 2).to(device)
        self.softmax = nn.Softmax(dim=1)

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input_vector, hidden_state):

        combined = torch.cat((input_vector, hidden_state.to(device)), 1)

        z = torch.sigmoid(torch.mm(combined, self.W_z) + self.b_z)
        r = torch.sigmoid(torch.mm(combined, self.W_r) + self.b_r)

        combined_2 = torch.cat((input_vector, r * hidden_state), 1)
        h_hat = torch.tanh(torch.mm(combined_2, self.W_h) + self.b_h)
        hidden_state = (torch.ones([1, self.hidden_size], device=device) - z) * hidden_state + z * h_hat

        output = self.out_linear(hidden_state)
        pre_label = self.softmax(output)

        return hidden_state.clone().detach(), pre_label

def train_GRU(dataloader, dev_dataloader, device ,input_size=50, hidden_size=50, n_epochs = 20, learning_rate = 0.001):

    print("input_size", input_size)
    print("hidden_size", hidden_size)
    print("n_epochs", n_epochs)
    print("device", device)
    print("learning_rate", learning_rate)
    
    HIDDEN_SIZE = hidden_size
    INPUT_SIZE = input_size
    gru = BasicGRU(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(gru.parameters(), lr=learning_rate)
    
    best_acc = 0

    for epoch in range(1, n_epochs + 1):
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        running_loss = 0
        for embedding_tensors, label_tensor in dataloader:
            hidden_state = torch.zeros(1, HIDDEN_SIZE, device=device) # 1 X n_neurons
            cell_state = torch.zeros(1, HIDDEN_SIZE, device=device) # 1 X n_neurons
            if label_tensor == -1:
                continue
            optimizer.zero_grad() # Clears existing gradients from previous epoch
            for tensor in embedding_tensors[0]:
                tensor = torch.reshape(tensor, (1,-1))
                hidden_state, output = gru.forward(tensor.to(device), hidden_state)
            # print(output, label_tensor)
            loss = criterion(output, label_tensor.to(device))
            loss.backward(retain_graph=True) # Does backpropagation and calculates gradients
            optimizer.step()
            # for p in gru.parameters():
            #     p.data.add_(p.grad.data, alpha=-learning_rate)
            running_loss += loss.item()
        ave_loss = running_loss / len(dataloader)
        print("Loss: {:.4f}".format(ave_loss))
        print("Testing on dev set...")
        dev_acc = test_GRU(dev_dataloader, gru, device, hidden_size=hidden_size)
        if dev_acc >= best_acc:
            best_acc = dev_acc
            print("Saving best model")
            PATH = './model/model_best/GRU_model_'+str(hidden_size)+"_"+str(input_size)+'.pt'
            torch.save(gru, PATH) 
        

def test_GRU(dataloader, model, device, hidden_size=50):
    correct = 0
    count = 0
    for embedding_tensors, label_tensor in dataloader:
        hidden_state = torch.zeros([1, hidden_size], device=device) # 1 X n_neurons
        if label_tensor == -1:
            continue
        for tensor in embedding_tensors[0]:
            tensor = torch.reshape(tensor, (1,-1))
            hidden_state, output = model.forward(tensor.to(device), hidden_state)
        if(float(torch.argmax(output) -  label_tensor.to(device)) == 0):
            correct+=1
        count+=1
    print("ACC", correct/count)
    return correct/count