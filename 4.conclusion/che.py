import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

class FCN(nn.Module):
    "Defines a connected network"
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        ########################################################################
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        ########################################################################
        # self.layer_hidden = nn.Sequential(
        #     nn.Linear(N_HIDDEN,N_HIDDEN),
        #     nn.ReLU())
        # self.hidden=nn.ModuleList()
        # for i in range(N_LAYERS):
        #     self.hidden.append(self.layer_hidden)
        ########################################################################
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        x = self.fcs(x)
        ########################################################################
        x = self.fch(x)
        ########################################################################
        # for layer in self.hidden:
        #     x = layer(x)
        ########################################################################
        x = self.fce(x)
        return x


data=np.array([[0.016364971768961182, 7822, 16, 5],
 [0.016364971768961182, 56, 19, 12],
 [0.0032319031537678114, 8032, 16, 5],
 [0.016364971768961182, 7041, 13, 5],
 [0.026301123061570415, 8965, 11, 4],
 [0.0032319031537678114, 4517, 16, 10],
 [0.06788220796044521, 8965, 16, 6],
 [0.0032319031537678114, 1692, 19, 2],
 [0.06042037069554462, 5134, 15, 2],
 [0.06622266956172207, 8965, 7, 16],
 [0.08012630792913335, 7822, 16, 5],
 [0.08729858435480223, 8965, 19, 12],
 [0.016364971768961182, 1314, 16, 5],
 [0.0032319031537678114, 4205, 13, 5],
 [0.0032319031537678114, 8965, 11, 7],
 [0.0032319031537678114, 8965, 3, 10],
 [0.0032319031537678114, 8965, 16, 18],
 [0.0032319031537678114, 5134, 6, 2],
 [0.0032319031537678114, 5134, 19, 16],
 [0.06622266956172207, 2603, 15, 2]])
from torchsummary import summary as summary_

model = FCN(1,1,30,3)
print(summary_(model,(1,1)))
z=0
while z<len(data):
    asd=data[z]
    model = FCN(1,1,int(asd[-2]),int(asd[-1]))
    print(summary_(model,(1,1)))
    z+=1