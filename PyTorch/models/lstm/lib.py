import pickle
import torch
from wordSequence import Word2Sequence

# print('a')
ws = pickle.load(open('../lstm/ws.pkl', 'rb'))

# 超参数
max_len = 200
batch_size = 256
embedding_dim = 100
hidden_size = 128
num_layers = 2
bidriectional = True
dropout = 0.2

epochs = 15

device = torch.device('cuda:0')
