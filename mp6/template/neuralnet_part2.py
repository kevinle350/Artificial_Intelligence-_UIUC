# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester
# Modified by Joao Marques (jmc12) for the fall 2021 semester 

"""
This is the main entry point for MP3. You should only modify code
within this file and neuralnet_part1.py,neuralnet_leaderboard -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader

class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension
        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        # self.net = nn.Sequential(
        #     nn.Conv2d(3, 16, 3),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 14, 3),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.MaxPool2d(2),
        # )
        self.net1 = nn.Conv2d(3, 16, 3)
        self.net2 = nn.Conv2d(16, 64, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(2304, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32,out_size)
        self.drop = nn.Dropout()
        self.optimizer = optim.Adam(self.parameters(), lrate)
        
    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        x = torch.reshape(x,(-1,3,32,32))
        x = self.pool(F.relu(self.net1(x)))
        x = self.pool(F.relu(self.net2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
        raise NotImplementedError("You need to write this part!")
        return torch.ones(x.shape[0], 1)

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """
        self.optimizer.zero_grad()
        p = self.forward(x)
        loss = self.loss_fn(p, y)
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()
        raise NotImplementedError("You need to write this part!")
        return 0.0

def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param epochs: an int, the number of epochs of training
    @param batch_size: size of each batch to train on. (default 100)

    This method _must_ work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: list of total loss at the beginning and after each epoch.
            Ensure that len(losses) == epochs.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    lr = 0.0001
    loss_fn = nn.CrossEntropyLoss()
    in_size = train_set.shape[1]
    out_size = 4
    net = NeuralNet(lr, loss_fn, in_size, out_size)
    stan_train = (train_set-train_set.mean())/(train_set.std())
    stan_dev = (dev_set-dev_set.mean())/(dev_set.std())

    losses = []
    for i in range(epochs):
        loss = []
        curr = 0
        while curr < len(stan_train):
            values = stan_train[curr:min(curr+batch_size, len(stan_train))]
            labels = train_labels[curr:min(curr+batch_size, len(stan_train))]
            curr_loss = net.step(values, labels)
            loss.append(curr_loss)
            curr += batch_size
        avg = np.mean(loss)
        losses.append(avg)

    y_hat = []
    predict = net(stan_dev)
    for i in range(len(predict)):
        y_hat.append(torch.argmax(predict[i]))
    
    return losses, np.array(y_hat).astype(np.int), net
    raise NotImplementedError("You need to write this part!")
    return [],[],None