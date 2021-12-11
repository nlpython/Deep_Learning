import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from dataset import get_data_loader
from wordSequence import Word2Sequence
import pickle
from lib import ws


max_len = 20
# print(ws.dict)
# print(len(ws))

class IMDB(nn.Module):

    def __init__(self):
        super(IMDB, self).__init__()
        self.embedding = nn.Embedding(len(ws), 100)
        self.fc1 = nn.Linear(max_len * 100, 2)

    def forward(self, input):
        """
        :param input: [batch_size, max_len]
        :return:
        """
        # [batch_size, max_len] => [batch_size, max_len, 100]
        x = self.embedding(input)
        # print('x:', x.size())
        x = x.view(128, -1)
        # print(x.size(
        out = self.fc1(x)

        return out


network = IMDB()
optimizer = optim.Adam(network.parameters(), lr=0.001)
def train(epochs):
    for epoch in range(epochs):
        for idx, (input, target) in enumerate(get_data_loader(True)):
            # 梯度归零
            optimizer.zero_grad()
            output = network(input)

            criteria = nn.CrossEntropyLoss()
            loss = criteria(output, target)
            loss.backward()
            optimizer.step()

            if idx % 10 == 0:
                print(f'epoch: {epoch+1} | step: {idx+1} | loss: {loss.item()}')

def test():
    test_loss = 0
    correct = 0
    mode = False
    network.eval()
    test_dataloader = get_data_loader(mode)
    with torch.no_grad():
        for input, target in test_dataloader:
            output = network(input)
            test_loss += F.nll_loss(output, target, reduction='sum')
            pred = torch.max(output, dim=-1)[-1]
            correct = pred.eq(target.data).sum()
        test_loss = test_loss / len(test_dataloader.dataset)
        print(f'loss: {test_loss}, acc: {correct / len(test_dataloader.dataset)}')

if __name__ == '__main__':
    # print(network)
    train(2)
    test()









