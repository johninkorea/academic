## imports
import os
import time as T
import numpy as np
import matplotlib.pyplot as plt

import torch

import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

## device choice
device = 'mps' if torch.backends.mps.is_built()  else 'cpu'
device = 'cpu'

# range of boundary
N=20
x_data=np.linspace(-1,1,N)
t_data=np.linspace(0,1,N)

# create boundary data list
xt_bnd_data = []
tu_bnd_data = []
for x in x_data:
    xt_bnd_data.append([x, 0])
    tu_bnd_data.append([-np.sin(np.pi * x)])
for t in t_data:
    xt_bnd_data.append([1, t])
    tu_bnd_data.append([0])
    xt_bnd_data.append([-1, t])
    tu_bnd_data.append([0])
xt_bnd_data = np.array(xt_bnd_data)
tu_bnd_data = np.array(tu_bnd_data)
plt.scatter(xt_bnd_data[:,0],xt_bnd_data[:,1],c='b')
# plt.scatter(tu_bnd_data[:,0],tu_bnd_data[:,1],c='r')



N=100
x_col_data = np.random.uniform(-1, 1, [N, 1])
t_col_data = np.random.uniform(0, 1, [N, 1])
xt_col_data = np.concatenate([x_col_data, t_col_data], axis=1)
print(np.shape(xt_col_data))
# xt_col_data = np.concatenate((xt_col_data, xt_bnd_data), axis=0)
# print(np.shape(xt_col_data))

plt.xlabel('x')
plt.ylabel('t')
plt.plot(xt_col_data[:,0],xt_col_data[:,1],"+",c='g', alpha=.1)
plt.show()

# define NN
class NN(nn.Module):
        def __init__(self): #, t, x): # 일단 hyper 안받는 걸로
            self.num1 = 10#t # nodes per hidden layer
            self.num2 = 10## number of hidden layer
            # self.batch=batch # batch size
            
            super(NN, self).__init__()
            self.layer_in = nn.Sequential(
                nn.Linear(1,self.num1),
                nn.ReLU())
            self.layer_hidden = nn.Sequential(
                nn.Linear(self.num1,self.num1),
                nn.ReLU())
            self.layer_out = nn.Sequential(
                nn.Linear(self.num1,1),
                nn.ReLU()) # 이거로 하면 바로 0으로 가서 학습하는게 의미 없어짐
                # nn.Sigmoid()) # 20번 학습하면 1e-9 order로 수렴함

            self.hidden=nn.ModuleList()
            for i in range(self.num2):
                self.hidden.append(self.layer_hidden)
            # self.fc1 = nn.Linear(1, 1)
            

        def forward(self, x): # 순서대로 대입해서 출력하는 것뿐 
            out = self.layer_in(x)
            for layer in self.hidden:
                out = layer(out)
            out = self.layer_out(out)
            # out=self.fc1(out)
            # out = out.view(10, 10)
            return out


model=NN().to(device)
def N_nn(t,x):
    re=model(t,x)
    return re
def N_gradient(t,x): # 이부분은 어떤 과정을 학습을 하는 거지?
    u=N_nn(t,x)
    u_t=np.gradient(u,t)
    u_x=np.gradient(u,x)
    u_xx=np.gradient(u_x,x)
    re= u_t+u*u_x-(.01/np.pi)*u_xx
    return re
def Loss(t,x):
    n=np.shape(t)[0]
    # # L1=nn.MSELoss(N_nn(t,x)-(real value를 어케 구해야해???))/n
    # # L2=nn.MSELoss(N_gradient(t,x)-(real value를 어케 구해야해???))/n
    # return L1+L2+.1*lamda


criterion = Loss(t,x).to(device)    # 비용 함수에 소프트맥스 함수 포함되어져 있음.
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

costh=[]
for epoch in range(epochs):
    avg_cost = 0

    for X, Y in train: # 미니 배치 단위로 꺼내온다. X는 이미지, Y는 레이블.
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        # print(hypothesis)
        # print(Y)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch
    avg_cost=avg_cost.detach().numpy()
    print('[Epoch: {:>2}] cost = {:>.6}'.format(epoch + 1, avg_cost))
    os.system(f"say {epoch + 1} epoch done {epoch + 1} epoch done")
    costh.append(avg_cost)













# def nnn(N,hyper):
#     ## set seed
#     seed=1
#     torch.manual_seed(seed)
#     np.random.seed(seed)

#     ## create data
#     N=int(N) # number of data
#     x=np.linspace(.001,40,N)
#     y=np.sin(x)/x

#     idx_train=(np.random.choice(N, int(N*.8), replace=0))
#     # idx_test = np.setdiff1d(np.arange(N), idx_train)

#     x_train=x[idx_train]
#     y_train=y[idx_train]
#     # x_test=x[idx_test]
#     # y_test=y[idx_test]

#     plt.scatter(x_train,y_train,s=1,c='r')
#     # plt.scatter(x_test,y_test,s=1,c='b')
#     # plt.show()

#     ## device choice
#     # device = 'mps' if torch.backends.mps.is_built()  else 'cpu'
#     device = 'cpu'
#     # # GPU 사용 가능일 경우 랜덤 시드 고정
#     # if device == 'mps':
#     #     # torch.backends.mps.manual_seed_all(seed)
#     #     torch.backends.mps.is_available()
#     # print("learning with",device)

#     # 신경망 정의
#     class NN(nn.Module):
#         def __init__(self, num1, num2):#, batch):
#             self.num1 = num1 # nodes per hidden layer
#             self.num2 = num2 # number of hidden layer
#             # self.batch=batch # batch size

#             super(NN, self).__init__()
#             self.layer_in = nn.Sequential(
#                 nn.Linear(1,self.num1),
#                 nn.ReLU())
#             self.layer_hidden = nn.Sequential(
#                 nn.Linear(self.num1,self.num1),
#                 nn.ReLU())
#             self.layer_out = nn.Sequential(
#                 nn.Linear(self.num1,1),
#                 # nn.ReLU()) # 이거로 하면 바로 0으로 가서 학습하는게 의미 없어짐
#                 nn.Sigmoid()) # 20번 학습하면 1e-9 order로 수렴함

#             self.hidden=nn.ModuleList()
#             for i in range(self.num2):
#                 self.hidden.append(self.layer_hidden)
#             # self.fc1 = nn.Linear(1, 1)
            

#         def forward(self, x): # 순서대로 대입해서 출력하는 것뿐 
#             out = self.layer_in(x)
#             for layer in self.hidden:
#                 out = layer(out)
#             out = self.layer_out(out)
#             # out=self.fc1(out)
#             # out = out.view(10, 10)
#             return out

    
#     ## set hyper parameter
#     # print(hyper)
#     learingrate, number_of_epoch, nodes_per_hidden, number_of_hidden = hyper
    
#     ## model define
#     lr=learingrate
#     epochs=number_of_epoch
#     batch_size = 1

#     nodes_per_hidden_layer=nodes_per_hidden
#     number_of_hidden_layer=number_of_hidden
#     model = NN(nodes_per_hidden_layer,number_of_hidden_layer).to(device)

#     optimizer = opt.Adam(model.parameters(), lr=lr)
#     criterion = torch.nn.MSELoss().to(device)


#     ## set data
#     from torch.utils.data import TensorDataset, DataLoader
#     # array to tensor
#     train_data = torch.Tensor(x_train)
#     train_label = torch.LongTensor(y_train) #-->이거를 long으로 하니까 loss가 더 빠르게 내려가
#     ds_train = TensorDataset(train_data, train_label)
#     train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
#     total_batch = len(train)

#     # test_data = torch.Tensor(x_test)
#     # test_label = torch.LongTensor(y_test)
#     # ds_test = TensorDataset(test_data, test_label)
#     # test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

#     # print(f"train stat: {hyper}")
#     ## train
#     costh=[]
#     for epoch in range(epochs):
#         avg_cost = 0
#         for X, Y in train: # 미니 배치 단위로 꺼내온다. X는 이미지, Y는 레이블.
#             X = X.to(device)
#             Y = Y.to(device)

#             optimizer.zero_grad()
#             hypothesis = model(X)

#             cost = criterion(hypothesis.to(torch.float32), Y.to(torch.float32))
#             cost.backward()
#             optimizer.step()

#             avg_cost += cost / total_batch
#         # print(avg_cost.item())
#         # print(type(avg_cost))
#         # avg_cost=avg_cost.detach().numpy()
#         avg_cost=avg_cost.item()
#         # print('[Epoch: {:>2}] cost = {:>.6}'.format(epoch + 1, avg_cost))
#         costh.append(avg_cost)


#     # plt.plot(range(epochs),costh)
#     # plt.show()

#     # 학습이 너무 잘되는 거라면 model을 실젤 그려 봐서 학습이 너무 잘되는 것이 맞는지 보자

#     # print("train done")
#     return costh[-1]
