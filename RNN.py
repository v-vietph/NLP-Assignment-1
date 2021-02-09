import torch
import numpy as np
from torch import nn

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class BasicRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size = 50, device='cpu'):
        super(BasicRNN, self).__init__()
        self.i2o = nn.Linear(hidden_size, 2).to(device)
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size).to(device)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input_vector, hidden_state):
        combined = torch.cat((input_vector, hidden_state), 1)
        hidden_state = self.i2h(combined)
        output = self.i2o(hidden_state)
        pre_label = self.softmax(output)
        return hidden_state.clone().detach(), pre_label

def train_RNN(dataloader, dev_dataloader, device, input_size=50, hidden_size=50, n_epochs = 20, learning_rate = 0.001):

    print("input_size", input_size)
    print("hidden_size", hidden_size)
    print("n_epochs", n_epochs)
    print("device", device)
    print("learning_rate", learning_rate)

    rnn = BasicRNN(input_size=input_size, hidden_size=hidden_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate) 
    best_acc = 0

    for epoch in range(1, n_epochs + 1):
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        running_loss = 0
        for embedding_tensors, label_tensor in dataloader:
            hidden_state = torch.rand(1, hidden_size, device=device) 
            if label_tensor == -1:
                continue
            optimizer.zero_grad() 
            for tensor in embedding_tensors[0]:
                tensor = torch.reshape(tensor, (1,-1))
                hidden_state, output = rnn.forward(tensor.to(device), hidden_state.to(device))
            loss = criterion(output, label_tensor.to(device))
            loss.backward(retain_graph=True) 

            optimizer.step()
            # for p in rnn.parameters():
            #     p.data.add_(p.grad.data, alpha=-learning_rate)
            running_loss += loss.item()
        ave_loss = running_loss / len(dataloader)
        print("Loss: {:.4f}".format(ave_loss))
        print("Testing on dev set...")
        dev_acc = test_RNN(dev_dataloader, rnn, device, hidden_size=hidden_size)
        if dev_acc >= best_acc:
            best_acc = dev_acc
            print("Saving best model with dev_acc", dev_acc)
            PATH = './model/model_best/RNN_model_'+str(hidden_size)+"_"+str(input_size)+'.pt'
            torch.save(rnn, PATH) 
       
def test_RNN(dataloader, model, device, hidden_size=50):
    correct = 0
    count = 0
    for embedding_tensors, label_tensor in dataloader:
        hidden_state = torch.zeros([1, hidden_size], device=device) # 1 X n_neurons
        if label_tensor == -1:
            continue
        for tensor in embedding_tensors[0]:
            tensor = torch.reshape(tensor, (1,-1))
            hidden_state, output = model.forward(tensor.to(device), hidden_state.to(device))
        if(float(torch.argmax(output) -  label_tensor.to(device)) == 0):
            correct+=1
        count+=1
    print("ACC", correct/count)
    return correct/count



