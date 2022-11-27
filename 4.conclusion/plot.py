import os
import numpy as np
import matplotlib.pyplot as plt


## log의 제일 좋은 결과가 나온 곳
# Generation = 15*********************************
# [[0.016364971768961182 7822 16 5]
#  [0.016364971768961182 56 19 12]
#  [0.0032319031537678114 8032 16 5]
#  [0.016364971768961182 7041 13 5]
#  [0.026301123061570415 8965 11 4]
#  [0.0032319031537678114 4517 16 10]
#  [0.06788220796044521 8965 16 6]
#  [0.0032319031537678114 1692 19 2]
#  [0.06042037069554462 5134 15 2]
#  [0.06622266956172207 8965 7 16]
#  [0.08012630792913335 7822 16 5]
#  [0.08729858435480223 8965 19 12]
#  [0.016364971768961182 1314 16 5]
#  [0.0032319031537678114 4205 13 5]
#  [0.0032319031537678114 8965 11 7]
#  [0.0032319031537678114 8965 3 10]
#  [0.0032319031537678114 8965 16 18]
#  [0.0032319031537678114 5134 6 2]
#  [0.0032319031537678114 5134 19 16]
#  [0.06622266956172207 2603 15 2]]
# Fitness    = 34349893.94037463
# Change     = 34187965.176504366


data_log=np.loadtxt("Fitness_log",dtype=float).T
# data_log=data_log[:20]
loss_log=(1/data_log)-0.00000001
print(np.min(loss_log))
plt.plot(np.arange(len((data_log))), loss_log,label="log", alpha=.3)

data_log_100gen=np.loadtxt("Fitness_log_100gen",dtype=float).T
# # data_log_100gen=data_log_100gen[:20]
loss_log_100gen=(1/data_log_100gen)-0.00000001
print(np.min(loss_log_100gen))
# plt.plot(np.arange(len((data_log_100gen))), loss_log_100gen,label="log_100gen", alpha=.3)

data_log2=np.loadtxt("Fitness_log2",dtype=float).T
# # data_log2=data_log2[:20]
loss_log2=(1/data_log2)-0.00000001
print(np.min(loss_log2))
plt.plot(np.arange(len((data_log2))), loss_log2,label="log2", alpha=.3)




plt.ylim([-0.000001,2e-5])
plt.legend()
plt.show()














