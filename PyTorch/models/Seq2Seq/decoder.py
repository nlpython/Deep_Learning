"""
解码器
"""
import torch
import torch.nn as nn
import lib
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from dataset import get_data_loader
import torch.nn.functional as F

class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(lib.ns), embedding_dim=lib.embedding_dim,
                                      padding_idx=lib.ns.PAD)
        self.gru = nn.GRU(input_size=lib.embedding_dim, hidden_size=lib.hidden_size, batch_first=True,
                          num_layers=lib.num_layers)
        self.fc = nn.Linear(lib.hidden_size, len(lib.ns))


    def forward(self, target, encoder_hidden):
        # 1.获取encoder的输出, 作为隐藏状态
        decoder_hidden = encoder_hidden
        # 2.准备decoder第一个时间步上的输入, [batch_size, 1] SOS作为输入
        decoder_input = torch.LongTensor(torch.ones([target.size(0), 1], dtype=torch.int64) * lib.ns.SOS).to(lib.device)
        # 3.在第一个时间步上进行计算, 得到第一个时间步的输出, hidden_state

        # 预测结果  [batch_size, max_len+2, vocab_size]
        decoder_outputs = torch.zeros([lib.batch_size, lib.max_len + 2, len(lib.ns)]).to(lib.device)
        for t in range(lib.max_len + 2):
            # decoder_output_t: [batch_size, vocab_size]
            decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs[:, t :] = decoder_output_t.unsqueeze(1)

            value, index = torch.topk(decoder_output_t, 1)
            decoder_input = index    #[batch_size, 1]

        return decoder_outputs, decoder_hidden


    def forward_step(self, decoder_input, decoder_hidden):
        """
        计算每个时间步上的输出
        :param decoder_input: [batch_size, 1]
        :param decoder_hidden: [1, batch_size, hidden_size]
        :return:
        """
        decoder_input_embedding = self.embedding(decoder_input) # [batch_size, 1, embedding_size]

        # out: [batch_size, 1, hidden_size]
        # decoder_hidden: [1, batch_size, hidden_size]
        out, decoder_hidden = self.gru(decoder_input_embedding)

        out = out.squeeze(1) # [batch_size, hidden_size]
        output = F.log_softmax(self.fc(out), dim=-1)
        # output: [batch_size, vocab_size]
        return output, decoder_hidden

if __name__ == '__main__':
    from encoder import Encoder
    decoder = Decoder()
    encoder = Encoder()
    for input, target, input_size, target_size in get_data_loader():
        out, encoder_hidden, _ = encoder(input, input_size)
        decoder(target, encoder_hidden)







