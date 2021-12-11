"""
Seq2Seq
"""
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
import lib
from torch.optim import Adam
from dataset import get_data_loader
import torch.nn.functional as F

class Seq2Seq(nn.Module):

    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, input, target, input_length, target_length):
        encoder_outputs, encoder_hidden = self.encoder(input, input_length)
        decoder_outputs, decoder_hidden = self.decoder(target, encoder_hidden)
        return decoder_outputs, decoder_hidden

net = Seq2Seq().to(lib.device)
optimizer = Adam(net.parameters(), lr=lib.learning_rate)
data_loader = get_data_loader(True)

def train(epochs):

    for epoch in range(epochs):
        for idx, (input, target, input_size, target_size) in enumerate(data_loader):
            # 调用cuda
            input, target, input_size, target_size = input.to(lib.device), target.to(lib.device), \
                                                     input_size.to(lib.device), target_size.to(lib.device)

            optimizer.zero_grad()
            decoder_outputs, _ = net(input, target, input_size, target_size)
            decoder_outputs = decoder_outputs.view(decoder_outputs.size(0)*decoder_outputs.size(1), -1) # [batch_size*seq_len, -1]
            target = target.view(-1) # [batch_size*seq_len]
            loss = F.nll_loss(decoder_outputs, target)
            loss.backward()
            optimizer.step()

            if idx % 20 == 0:
                print(f'epoch: {epoch+1} / {epochs} | step: {idx} | loss: {loss.item()}')


if __name__ == '__main__':
    train(lib.epochs)

