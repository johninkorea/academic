import torch

# 데이터섹을 클래스로 커스텀해보자
# 기본 구성
class Customdataset(torch.utils.data.Dataset):
    def __init__(self): # 데이터섹을 전처리
        return
    
    def __len__(self): # 데이터섹을 총 샘플의 수를 적는 부분
        return
    
    def __getitem__(self, idx): # 데이터셋에서 특정 하나를 가져오는 함수
        return

#############################################################################
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

torch.manual_seed(1)

class Customdataset(torch.utils.data.Dataset):
    def __init__(self, x_data, y_data): # 데이터섹을 전처리
        self.x_data= x_data
        self.y_data= y_data

    def __len__(self): # 데이터섹을 총 샘플의 수를 적는 부분
        return len(self.x_data)
    
    def __getitem__(self, idx): # 데이터셋에서 특정 하나를 가져오는 함수
        x=torch.FloatTensor(self.x_data[idx])
        y=torch.FloatTensor(self.y_data[idx])
        return x,y

# 학습 시킬 데이터를 받고, 그러를 함수로 묶고 정리해서 _train으로 만든다.
x_data=[[73, 80, 75],
                   [93, 88, 93],
                   [89, 91, 90],
                   [96, 98, 100],
                   [73, 66, 70]]
y_data=[[152], [185], [180], [196], [142]]
dataset=Customdataset(x_data,y_data)
dataloader=DataLoader(dataset,batch_size=2, shuffle=1)

model=torch.nn.Linear(3,1)
opt_m=torch.optim.SGD(model.parameters(),lr=1e-5)

cos_histroy=[]
epochs=20
for epoch in range(epochs+1):
    for batch_idx, sample in enumerate(dataloader):
        x_train,y_train=sample

        prediction=model(x_train)
        cos=F.mse_loss(prediction,y_train)

        opt_m.zero_grad()
        cos.backward()
        opt_m.step()

        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
        epoch, epochs, batch_idx+1, len(dataloader),cos.item()
        ))
    cos_histroy.append(cos.item())

import matplotlib.pyplot as plt
plt.plot(range(epochs+1),cos_histroy)
plt.xlim([5,21])
plt.ylim([0,3])
plt.show()


# 이번에 도 들죽날쭋하네
# 셔플 텨져있음.

in_val=torch.FloatTensor([73,80,75])
out_val=model(in_val)
print(out_val.item())


