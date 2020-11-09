# coding: utf-8

import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # ,cross_val_score,KFold
import matplotlib.pyplot as plt

begin_time = time.time()

trainingSet = pd.read_csv('trainingSet.csv', header=None)
testSet = pd.read_csv('testSet.csv', header=None)

# 导入数据
X = trainingSet.iloc[:, 0:112]
y = trainingSet.iloc[:, 115:118]  # concentration
test_X = testSet.iloc[:, 0:112]
test_y = testSet.iloc[:, 115:118]

# covert data to tensor
train_X = np.array(X)
train_X = torch.from_numpy(train_X)
train_X = train_X.float().cuda()
# print(train_X.dtype)
train_y = np.array(y)
train_y = torch.from_numpy(train_y)
train_y = train_y.float().cuda()
test_X = np.array(test_X)
test_X = torch.from_numpy(test_X)
test_X = test_X.float().cuda()
test_y = np.array(test_y)
test_y = torch.from_numpy(test_y)
test_y = test_y.float().cuda()


# construct nn
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden3 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden4 = torch.nn.Linear(n_hidden, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))# activation function for hidden layer
        x = self.predict(x)  # linear output
        return x


# 15 75% | 20 75% |
net = Net(n_feature=112, n_hidden=15, n_output=3)  # define the network
net.cuda()
print(net)  # net architecture

# optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
optimizer = torch.optim.Adam(net.parameters(), lr=0.2, betas=(0.9, 0.99))
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
#loss_func = torch.nn.L1Loss()
loss_his = []
accuracy_his = []
train_accuracy_his = []
test_loss_his = []
# plt.ion()   # something about plotting

for t in range(300):
    prediction = net(train_X)  # input x and predict based on x
    train_accuracy = (prediction - train_y)/prediction
    train_accuracy = 1 - torch.mean(torch.abs(torch.tensor(train_accuracy)))


    test_prediction = net(test_X)
    accuracy = (test_prediction - test_y) / test_prediction
    accuracy = 1 - torch.mean(torch.abs(torch.tensor(accuracy)))

    # print(accuracy.size())
    loss = loss_func(prediction, train_y)  # must be (1. nn output, 2. target)
    test_loss = loss_func(test_prediction, test_y)
    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients

    print('train_loss:', str(loss.item()))
    cpu_loss = loss.cpu()
    loss_his.append(cpu_loss.data.numpy())
    print('test_loss:', str(test_loss.item()))
    cpu_test_loss = test_loss.cpu()
    test_loss_his.append(cpu_test_loss.data.numpy())
    print('Accuracy:', str(accuracy.item()))
    cpu_accuracy = accuracy.cpu()
    accuracy_his.append(cpu_accuracy.data.numpy())
    cpu_train_accuracy = train_accuracy.cpu()
    train_accuracy_his.append(cpu_train_accuracy.data.numpy())

    #print(test_prediction)

# train loss plot
plt.figure()
plt.plot(loss_his)
plt.xlabel('Steps')
plt.ylabel('Train Loss')
plt.savefig('./results/concentration_net_train_loss.png')
np.save('./results/concentration_train_loss_his.npy', loss_his)

# Test loss plot
plt.figure()
plt.plot(test_loss_his)
plt.xlabel('Steps')
plt.ylabel('Test Loss')
plt.savefig('./results/concentration_net_test_loss.png')
np.save('./results/concentration_test_loss_his.npy', test_loss_his)

# test accuracy plot
plt.figure()
plt.plot(accuracy_his)
plt.xlabel('Steps')
plt.ylabel('Accuracy')
plt.savefig('./results/concentration_net_test_accuracy.png')
np.save('./results/concentration_test_accuracy.npy', accuracy_his)

# train accuracy plot
plt.figure()
plt.plot(train_accuracy_his)
plt.xlabel('Steps')
plt.ylabel('Accuracy')
plt.savefig('./results/concentration_net_train_accuracy.png')
np.save('./results/concentration_train_accuracy.npy', train_accuracy_his)

torch.save(net, ('./results/concentration_net.pkl'))

# Benchmark vs Error plot
print(test_y)
print(test_prediction)
