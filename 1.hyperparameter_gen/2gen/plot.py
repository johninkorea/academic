import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def save_gif_PIL(outfile, files, fps=5, loop=0):
    "Helper function for saving GIFs"
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format="GIF", append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)

data=np.loadtxt("fit_log",dtype=float).T
data=np.copy(data[:450])

list=np.sort([0,11,113,115,13,16,163,165,175,177,18,2,20,200,202,24,26,3,4,427,429,6,8])
list.tolist()
print(list)
print(len(list))


import os
os.system("mkdir plots")
files = []
z=0
for i in range(len(data)):
    # plot the result as training progresses
    # if (i+1) % 3 == 0: 
    if  i in list: 
        print(z)
        file = f"plots/fitness_{i}.png"
        asd=np.copy(data[:i])
        plt.plot(np.log10(np.arange(len((asd)))), np.exp(-asd), alpha=.7,c='b')
        plt.xlabel("Generation(log scale)",size=15)
        plt.ylabel("exp(-Fitness)",size=15)
        # print(np.min(loss_log2))

        plt.ylim([-.1,.9])
        plt.xlim([0,(np.log10(650))])
        # plt.legend()
        plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=300, facecolor="white")
        plt.cla()
        files.append(file)
        z+=1
z=0
while z<8:
    files.append(files[-1])
    z+=1
print(10000000000)
save_gif_PIL("img/ggg.gif", files, fps=2, loop=0)












