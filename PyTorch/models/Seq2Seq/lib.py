from Num_sequence import Num_sequnece

ns = Num_sequnece()
max_len = 9
batch_size = 128
embedding_dim = 100
num_layers = 2
hidden_size = 12
dropout = 0.5

epochs = 10
learning_rate = 0.0001

import torch
device = torch.device('cuda:0')