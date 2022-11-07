import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from NN.nn import nnn

import numpy as np


# hyper=np.array([learingrate, number_of_epoch, nodes_per_hidden, number_of_hidden])
hyper=np.array([1e-2, 10, 12, 3],dtype=str)
asd=nnn(hyper)

# 불러 학습하기 성공