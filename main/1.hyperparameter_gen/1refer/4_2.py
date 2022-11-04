import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

torch.manual_seed(1)

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

model=nn.Sequential(
    nn.Linear(2,1),
    nn.Sigmoid()
)

a=model(x_train)
print(a)


opt_m=opt.SGD(model.parameters(), lr=1)

epochs=int(1e3)

for epoch in range(epochs):
    hy=model(x_train)
    cos=F.binary_cross_entropy(hy,y_train)

    opt_m.zero_grad()
    cos.backward()
    opt_m.step()

    if epoch%20==0:
        prediction=model(x_train)>torch.FloatTensor([.5])
        corret= (prediction.float()==y_train)
        accuracy=corret.sum().item()/len(corret)

        print('epoch {:4d}/{} Cost: {:.6f}  accuracy: {:2.2f}'.format(epoch, epochs, cos.item(), accuracy))


a=model(x_train)
print(a)
print(list(model.parameters()))

