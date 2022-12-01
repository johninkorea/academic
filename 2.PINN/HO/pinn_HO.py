import numpy as np
import os
import torch
import torch.nn as nn

from PIL import Image
import matplotlib.pyplot as plt

seed=1114
np.random.seed(seed)
torch.manual_seed(seed)


def save_gif_PIL(outfile, files, fps=5, loop=0):
    "Helper function for saving GIFs"
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)

def plot_result(x,y,x_data,y_data,yh,xp=None):
    "Pretty plot training results"
    plt.figure(figsize=(8,5))
    plt.plot(x,y, color="gray", linewidth=2, alpha=0.8, label="Exact solution")
    plt.scatter(x_data, y_data, marker="s", color="r", alpha=0.8, label='Training data')
    # print(xp)
    if xp is not None:
        qwe=xp.numpy().T[0]
        qwe2=y.T[0]
        # print(qwe2)
        plt.scatter(x[::25],y[::25], marker="^", color="k", alpha=1, label='Gradient checkpoint')
        # plt.scatter(xp, qwe2[qwe],)
    plt.plot(x,yh, color="tab:blue", linewidth=4, alpha=0.8, label="Neural network prediction")
    plt.legend(loc='lower right', prop={'size': 10})
    plt.xlabel("Time", size=20)
    plt.ylabel("Displacement", size=20)
    plt.xlim(-0., 1.)
    plt.ylim(-1.1, 1.1)
    plt.text(.6,0.8,"Epoch: %i"%(i+1),fontsize="xx-large",color="k")
    # plt.axis("off")
def oscillator(d, w0, x):
    """Defines the analytical solution to the 1D underdamped harmonic oscillator problem. 
    Equations taken from: https://beltoforion.de/en/harmonic_oscillator/"""
    assert d < w0
    w = np.sqrt(w0**2-d**2)
    phi = np.arctan(-d/w)
    A = 1/(2*np.cos(phi))
    cos = torch.cos(phi+w*x)
    sin = torch.sin(phi+w*x)
    exp = torch.exp(-d*x)
    y  = exp*2*A*cos
    return y
class FCN(nn.Module):
    "Defines a connected network"
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


## create data to train
d, w0 = 1.5, 15
# get the analytical solution over the full domain
x = torch.linspace(0,1,500).view(-1,1) # 0차원을 1차원으로 만들어 주는
y = oscillator(d, w0, x).view(-1,1)
print(x.shape, y.shape)

# slice out a small number of points from the LHS of the domain
index=np.random.choice(np.arange(len(x)), replace=0, size=10)
x_data = x[index]
y_data = y[index]
print(x_data.shape, y_data.shape)




## pinn
x_physics = torch.linspace(0,1,20).view(-1,1).requires_grad_(True)# sample locations over the problem domain
mu, k = 2*d, w0**2
os.system("mkdir plots")

model = FCN(1,1,30,3)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
files = []
losslist=[]
for i in range(15000):
    optimizer.zero_grad()
    
    # compute the "data loss"
    yh = model(x_data)
    loss1 = torch.mean((yh-y_data)**2)# use mean squared error
    
    # compute the "physics loss"
    yhp = model(x_physics)
    dx  = torch.autograd.grad(yhp, x_physics, torch.ones_like(yhp), create_graph=True)[0]# computes dy/dx
    dx2 = torch.autograd.grad(dx,  x_physics, torch.ones_like(dx),  create_graph=True)[0]# computes d^2y/dx^2
    physics = dx2 + mu*dx + k*yhp# computes the residual of the 1D harmonic oscillator differential equation
    loss2 = (1e-4)*torch.mean(physics**2)
    
    # backpropagate joint loss
    loss = loss1 + loss2# add two loss terms together
    losslist.append(loss.detach().item())
    loss.backward()
    optimizer.step()
    
    
    # plot the result as training progresses
    if (i+1) % 100 == 0: 
        
        yh = model(x).detach()
        xp = x_physics.detach()
        
        plot_result(x,y,x_data,y_data,yh,xp)
        
        file = "plots/pinn_%.8i.png"%(i+1)
        plt.savefig(file, pad_inches=0.1, dpi=100, facecolor="white")
        files.append(file)
        
        # if (i+1) % 6000 == 0: plt.show()
        # else: plt.close("all")
print(f"Last loss: {losslist[-1]}")
z=0
while z<80:
    files.append(files[-1])
    z+=1

save_gif_PIL("result/pinn.gif", files, fps=20, loop=0)



os.system("say done")







