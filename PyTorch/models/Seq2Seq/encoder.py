"""
编码器
"""
import torch
import torch.nn as nn
import lib
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from dataset import get_data_loader

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(lib.ns), embedding_dim=lib.embedding_dim,
                                      padding_idx=lib.ns.PAD)   # 传入后PAD相应的权重将不参与更新
        self.gru = nn.GRU(input_size=lib.embedding_dim, num_layers=lib.num_layers, hidden_size=lib.hidden_size,
                          batch_first=True, dropout=lib.dropout)



    def forward(self, input, input_length):
        """
        :param input: [batch_size, max_len]
        :return:
        """
        embeded = self.embedding(input)
        # 打包, 加入gru计算
        input_length = torch.tensor(input_length, dtype=torch.int64, device=torch.device('cpu'))
        embeded = pack_padded_sequence(embeded, input_length, batch_first=True)
        out, hidden = self.gru(embeded)
        out, out_length = pad_packed_sequence(out, batch_first=True, padding_value=lib.ns.PAD)

        # out: [batch_size, seq_len, hidden_size]
        # hidden: [1 * 2, batch_size, hidden_size]
        return out, hidden


if __name__ == '__main__':
    encoder = Encoder()
    for input, target, input_len, target_len in get_data_loader(True):
        out, hidden, out_length = encoder(input, input_len)
        print(out.size())
        print(hidden.size())
        print(out_length)
        break




