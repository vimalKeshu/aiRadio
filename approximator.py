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

    def __init__(self, input_size, output_size=2, path=None):
        self.model = ApproximatorDNN(input_size=input_size, output_size=output_size).double().to(device)
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.RMSprop(self.model.parameters())
        self.epoch = 1
        if path:
            self.load(path=path)

    def train(self, x, y):
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        x = x.to(device)
        y = y.to(device)
        # feed forward
        x = self.model(x)
        # feed backward
        self.optimizer.zero_grad()
        self.loss = self.criterion(x, y)
        self.loss.backward()
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

    def load(self, path) -> None:
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']
        self.model.train()

    def save(self, path) -> None:
        torch.save({
        'epoch': self.epoch,
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'loss': self.loss
        }, path)