## https://github.com/ahmedfgad/TorchGA

## https://pygad.readthedocs.io/en/latest/index.html



---------------------------

num of hidden layers, 
activation function, 
num of epochs, batch size, 
learning rate with scheduler

---------------------------
activation function

lr=1e-2

epochs=20

batch_size = 1

nodes_per_hidden_layer=12

number_of_hidden_layer=2

optimizer = opt.Adam(model.parameters(), lr=lr)

criterion = torch.nn.MSELoss().to(device)


