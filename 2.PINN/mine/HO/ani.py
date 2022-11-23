import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def save_gif_PIL(outfile, files, fps=5, loop=0):
    "Helper function for saving GIFs"
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)

files = []

for i in range(15000):
    # plot the result as training progresses
    if (i+1) % 10 == 0: 
        
        file = f"plots/nn_{i+1}.png"
        files.append(file)
            
save_gif_PIL("result/nn.gif", files, fps=20, loop=0)

files = []
for i in range(15000):
    # plot the result as training progresses
    if (i+1) % 10 == 0: 

        file = "plots/pinn_%.8i.png"%(i+1)
        files.append(file)
save_gif_PIL("result/pinn.gif", files, fps=20, loop=0)











