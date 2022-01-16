import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ApproximatorDNN(nn.Module):

    def __init__(self, input_size, output_size):
        super(ApproximatorDNN, self).__init__()
        self.input_nuerons = input_size
        self.output_nuerons = output_size
        self.hidden_nuerons = int(self.input_nuerons/self.output_nuerons)
        self.layer_1 = nn.Linear(self.input_nuerons, self.hidden_nuerons)
        self.head = nn.Linear(self.hidden_nuerons, self.output_nuerons)

    def forward(self, x):
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.head(x)
        return x

class Approximator(object):

    def __init__(self, input_size, output_size=2):
        self.model = ApproximatorDNN(input_size=input_size, output_size=output_size).double().to(device)
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.RMSprop(self.model.parameters())

    def train(self, x, y):
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        x = x.to(device)
        y = y.to(device)
        # feed forward
        x = self.model(x)
        # feed backward
        self.optimizer.zero_grad()
        loss = self.criterion(x, y)
        loss.backward()
        self.optimizer.step()

    def predict(self, x):
        with torch.no_grad():
            x = torch.from_numpy(x)
            x = x.to(device)
            x = self.model(x)
            print(x.numpy())
            x = torch.max(x, 1)
            #x = x.numpy()
        return x

    def load(path):
        pass

    def store(path):
        pass