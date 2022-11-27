import os
import numpy as np
import matplotlib.pyplot as plt


data=np.loadtxt("asd",dtype=float).T

plt.plot(np.arange(len((data))), data)
plt.show()



