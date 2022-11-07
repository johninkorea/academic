## imports
import numpy as np
import matplotlib.pyplot as plt

import torch

import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def nnn(hyper):
    ## set seed
    seed=1
    torch.manual_seed(seed)
    np.random.seed(seed)

    ## create data
    N=int(1e1) # number of data
    x=np.linspace(.001,20,N)
    y=np.sin(x)/x

    idx_train=(np.random.choice(N, int(N*.8), replace=0))
    # idx_test = np.setdiff1d(np.arange(N), idx_train)

    x_train=x[idx_train]
    y_train=y[idx_train]
    # x_test=x[idx_test]
    # y_test=y[idx_test]

    plt.scatter(x_train,y_train,s=1,c='r')
    # plt.scatter(x_test,y_test,s=1,c='b')
    # plt.show()

    ## device choice
    # device = 'mps' if torch.backends.mps.is_built()  else 'cpu'
    device = 'cpu'
    # # GPU 사용 가능일 경우 랜덤 시드 고정
    # if device == 'mps':
    #     # torch.backends.mps.manual_seed_all(seed)
    #     torch.backends.mps.is_available()
    # print("learning with",device)

    # 신경망 정의
    class NN(nn.Module):
        def __init__(self, num1, num2):#, batch):
            self.num1 = num1 # nodes per hidden layer
            self.num2 = num2 # number of hidden layer
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
                # nn.ReLU()) # 이거로 하면 바로 0으로 가서 학습하는게 의미 없어짐
                nn.Sigmoid()) # 20번 학습하면 1e-9 order로 수렴함

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

    
    ## set hyper parameter
    # print(hyper)
    learingrate, number_of_epoch, nodes_per_hidden, number_of_hidden = hyper
    
    ## model define
    lr=learingrate
    epochs=number_of_epoch
    batch_size = 1

    nodes_per_hidden_layer=nodes_per_hidden
    number_of_hidden_layer=number_of_hidden
    model = NN(nodes_per_hidden_layer,number_of_hidden_layer).to(device)

    optimizer = opt.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss().to(device)


    ## set data
    from torch.utils.data import TensorDataset, DataLoader
    # array to tensor
    train_data = torch.Tensor(x_train)
    train_label = torch.LongTensor(y_train) #-->이거를 long으로 하니까 loss가 더 빠르게 내려가
    ds_train = TensorDataset(train_data, train_label)
    train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    total_batch = len(train)

    # test_data = torch.Tensor(x_test)
    # test_label = torch.LongTensor(y_test)
    # ds_test = TensorDataset(test_data, test_label)
    # test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    # print(f"train stat: {hyper}")
    ## train
    costh=[]
    for epoch in range(epochs):
        avg_cost = 0
        for X, Y in train: # 미니 배치 단위로 꺼내온다. X는 이미지, Y는 레이블.
            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()
            hypothesis = model(X)

            cost = criterion(hypothesis.to(torch.float32), Y.to(torch.float32))
            cost.backward()
            optimizer.step()

            avg_cost += cost / total_batch
        # print(avg_cost.item())
        # print(type(avg_cost))
        # avg_cost=avg_cost.detach().numpy()
        avg_cost=avg_cost.item()
        # print('[Epoch: {:>2}] cost = {:>.6}'.format(epoch + 1, avg_cost))
        costh.append(avg_cost)


    # plt.plot(range(epochs),costh)
    # plt.show()

    # print("train done")
    return costh[-1]
