import torch
import numpy as np
from torch import nn

class BasicRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size = 50):
        super(BasicRNN, self).__init__()
        self.i2o = nn.Linear(hidden_size, 2)
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input_vector, hidden_state):

        combined = torch.cat((input_vector, hidden_state), 1)
        hidden_state = self.i2h(combined)
        output = self.i2o(hidden_state)
        pre_label = self.softmax(output)
        return hidden_state.clone().detach(), pre_label

def train_RNN(dataloader, device, input_size=50, hidden_size=50, n_epochs = 20, learning_rate = 0.001):

    print("input_size", input_size)
    print("hidden_size", hidden_size)
    print("n_epochs", n_epochs)
    print("device", device)
    print("learning_rate", learning_rate)

    rnn = BasicRNN(input_size=input_size, hidden_size=hidden_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters())    
    for epoch in range(1, n_epochs + 1):
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        hidden_state = torch.zeros(1, hidden_size, device=device) 
        for embedding_tensors, label_tensor in dataloader:
            if label_tensor == -1:
                continue
            optimizer.zero_grad() 
            for tensor in embedding_tensors[0]:
                tensor = torch.reshape(tensor, (1,-1))
                hidden_state, output = rnn.forward(tensor.to(device), hidden_state)
            loss = criterion(output, label_tensor)
            loss.backward(retain_graph=True) 

            for p in rnn.parameters():
                p.data.add_(p.grad.data, alpha=-learning_rate)

        print("Loss: {:.4f}".format(loss.item()))

        PATH = './model/RNN_model'+str(hidden_size)+"-"+str(input_size)+"-"+str(epoch)+'.pt'
        torch.save(rnn, PATH) 

def test_RNN(dataloader, model, device, hidden_size=50):
    correct = 0
    count = 0
    for embedding_tensors, label_tensor in dataloader:
        hidden_state = torch.zeros(1, hidden_size) # 1 X n_neurons
        if label_tensor == -1:
            continue
        for tensor in embedding_tensors[0]:
            tensor = torch.reshape(tensor, (1,-1))
            hidden_state, output = model.forward(tensor.to(device), hidden_state)
        if(float(torch.argmax(output) -  label_tensor) == 0):
            correct+=1
        count+=1
    print("ACC", correct/count)



