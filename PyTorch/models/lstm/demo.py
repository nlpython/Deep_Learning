"""
lstm使用实例
"""
import torch
import torch.nn as nn

# 超参数
batch_size = 100
seq_len = 20
vocab_size = 100
embedding_dim = 30

hidden_size = 18
num_layer = 1

input = torch.randint(low=0, high=100, size=[batch_size, seq_len])

# 数据经过embedding处理
embeddiing = nn.Embedding(vocab_size, embedding_dim)
# => [batch_size, seq_len, embedding_dim]
input_embeded = embeddiing(input)

# 将input_embeded数据传入lstm
lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layer, batch_first=True)
output, (h_n, c_n) = lstm(input_embeded)

print(output)
print(output.size())    # [100, 20, 18]
print(h_n.size())       # [1, 100, 18]
print(c_n.size())       # [1, 100, 18]

# 获取最后一次时间步上输出
print(output[:, -1, :].size())   # [100, 18]
# 获取最后一次时间步上的hidden_state
print(h_n[-1:, :, :].size())     # [1, 100, 18]
