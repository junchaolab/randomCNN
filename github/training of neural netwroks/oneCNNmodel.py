# coding: utf-8
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import pandas as pd
# import torchvision
# from sklearn.model_selection import train_test_split#,cross_val_score,KFold
import matplotlib.pyplot as plt
import joblib

begin_time = time.time()
EPOCH = 200 # train the training data n times, to save time, we just train 1 epoch
# BATCH_SIZE = 50
LR = 0.00001  # learning rate
TRAIN_NUM = 8000

data = pd.read_csv('randomAll.csv', header=None)
# 导入数据
X = joblib.load('valve_map_all.pkl')
physics = 'VandC'
V = data.iloc[:, 112:115].to_numpy()  # velocity
#physics = 'concentration'
C = data.iloc[:, 115:118].to_numpy()  # concentration
Y = np.hstack(((V*50),C))

X = X.ravel()
X = X.reshape(10513, 1, 15, 15)

train_X = torch.from_numpy(X[0:TRAIN_NUM, :, :, :]).cuda()
train_y = torch.from_numpy(Y[0:TRAIN_NUM, :]).cuda()
test_X = torch.from_numpy(X[TRAIN_NUM:10513, :, :, :]).cuda()
test_y = torch.from_numpy(Y[TRAIN_NUM:10513, :]).cuda()

print(train_X.size())
print(train_y.size())
print(test_X.size())
print(test_y.size())


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  
            nn.Conv2d(
                in_channels=1,  
                out_channels=16,  
                kernel_size=5,  
                stride=1,  
                padding=2,
                
            ), 
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  
        )
        self.conv2 = nn.Sequential(  #
            nn.Conv2d(16, 32, 5, 1, 2),  
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  #
        )
        self.out = nn.Linear(288, 6)  # 

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  #
        output = self.out(x)
        return output#, x  # 


cnn = CNN()
cnn = cnn.cuda()
print(cnn)  # net architecture

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.MSELoss()
cnn = cnn.double()

# training
loss_his = []
train_accuracy_his = []
accuracy_his = []
test_loss_his = []

for epoch in range(EPOCH):
    output = cnn(train_X)
    loss = loss_func(output, train_y)
    train_acc = (output - train_y) / output
    train_acc = 1 - torch.mean(torch.abs(torch.tensor(train_acc)))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    test_prediction = cnn(test_X)
    test_loss = loss_func(test_prediction, test_y)
    accuracy = (test_prediction - test_y) / test_prediction
    accuracy = 1 - torch.mean(torch.abs(torch.tensor(accuracy)))

    print('EPOCH:', epoch)
    print('train_loss:', str(loss.item()))
    cpu_loss = loss.cpu()
    loss_his.append(cpu_loss.data.numpy())
    print('test_loss:', str(test_loss.item()))
    cpu_test_loss = test_loss.cpu()
    test_loss_his.append(cpu_test_loss.data.numpy())
    print('test Accuracy:', str(accuracy.item()))
    cpu_accuracy = accuracy.cpu()
    accuracy_his.append(cpu_accuracy.data.numpy())
    print('train Accuracy:', str(train_acc.item()))
    cpu_train_acc = train_acc.cpu()
    train_accuracy_his.append(cpu_train_acc.data.numpy())

    print(test_prediction)


# train loss plot
plt.figure()
plt.plot(loss_his)
plt.xlabel('Steps')
plt.ylabel('Train Loss')
plt.savefig('./' + physics + '/' + str(begin_time) + 'train_loss.png')
np.save('./' + physics + '/' + str(begin_time) + 'train_loss.npy', loss_his)

# Test loss plot
plt.figure()
plt.plot(test_loss_his)
plt.xlabel('Steps')
plt.ylabel('Test Loss')
plt.savefig('./' + physics + '/' + str(begin_time) + 'test_loss.png')
np.save('./' + physics + '/' + str(begin_time) + 'test_loss.npy', test_loss_his)

# accuracy plot
plt.figure()
plt.plot(accuracy_his)
plt.xlabel('Steps')
plt.ylabel('Accuracy')
plt.savefig('./' + physics + '/' + str(begin_time) + 'test_accuracy.png')
np.save('./' + physics + '/' + str(begin_time) + 'test_accuracy.npy', accuracy_his)

# accuracy plot
plt.figure()
plt.plot(train_accuracy_his)
plt.xlabel('Steps')
plt.ylabel('Accuracy')
plt.savefig('./' + physics + '/' + str(begin_time) + 'train_accuracy.png')
np.save('./' + physics + '/' + str(begin_time) + 'train_accuracy.npy', train_accuracy_his)

# save network
torch.save(cnn, ('./' + physics + '/' + str(begin_time) + 'net.pkl'))
