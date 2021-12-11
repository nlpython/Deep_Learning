import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from dataset import get_data_loader
from wordSequence import Word2Sequence
import pickle
import lib
from tqdm import tqdm



class LSTMIMDB(nn.Module):

    def __init__(self):
        super(LSTMIMDB, self).__init__()
        self.embedding = nn.Embedding(len(lib.ws), lib.embedding_dim)
        # 加入lstm层
        self.lstm = nn.LSTM(input_size=lib.embedding_dim, hidden_size=lib.hidden_size, num_layers=lib.num_layers, batch_first=True,
                            bidirectional=lib.bidriectional, dropout=lib.dropout)

        self.fc1 = nn.Linear(lib.hidden_size * 2, 10)
        self.drop = nn.Dropout(0.1)

    def forward(self, input):
        """
        :param input: [batch_size, max_len]
        :return:
        """
        # [batch_size, max_len] => [batch_size, max_len, 100]
        x = self.embedding(input)
        # x: [batch_size, max_len, 2 * hidden_size], h_n: [2 * 2, batch_size, hidden_size]
        x, (h_n, c_n) = self.lstm(x)
        # 获取两个方向最后一次的output, 进行concat
        output_f = h_n[-2, :, :]    # 正向最后一次输出
        output_b = h_n[-1, :, :]    # 反向最后一次输出
        output = torch.cat([output_f, output_b], dim=-1)    # [batch_size, hidden_size * 2]

        # [batch_size, 2]
        out = self.fc1(output)
        out = self.drop(out)

        return out


network = LSTMIMDB().to(lib.device)
optimizer = optim.Adam(network.parameters(), lr=0.001)
criteria = nn.CrossEntropyLoss()


def train(epochs):
    for epoch in range(epochs):

        correct, total = 0, 0
        for idx, (input, target) in enumerate(get_data_loader(True)):
            # 调用cuda
            input = input.to(lib.device)
            target = target.to(lib.device)

            # 梯度归零
            optimizer.zero_grad()
            output = network(input)

            loss = criteria(output, target)
            loss.backward()
            optimizer.step()

            pred = torch.max(output, dim=-1)[-1]
            correct += (pred == target).sum().item()
            total += output.size(0)

            if idx % 10 == 0:
                print(f'epoch: {epoch+1} / {epochs} | step: {idx+1} | loss: {loss.item()} | accuracy: {correct / total}')
                correct, total = 0, 0

def test():
    test_loss = 0
    correct = 0
    total = 0
    mode = False
    network.eval()
    test_dataloader = get_data_loader(mode)
    with torch.no_grad():
        for input, target in tqdm(test_dataloader, 'Testing'):
            input = input.to(lib.device)
            target = target.to(lib.device)
            output = network(input)

            criteria = nn.CrossEntropyLoss()
            test_loss += criteria(output, target)
            pred = torch.max(output, dim=-1)[-1]
            correct += (pred == target).sum().item()

            total += output.size(0)

        print('On test set:', f'loss: {test_loss / total}, accuracy: {correct / total}')

if __name__ == '__main__':
    # train(lib.epochs)
    # torch.save(network.state_dict(), './model/lstm.pkl')
    # torch.save(optimizer.state_dict(), './model/optim.pkl')
    #
    # 评估模型
    network.load_state_dict(torch.load('./model/lstm.pkl'))
    network = network.to(lib.device)
    test()










