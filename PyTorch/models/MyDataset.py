import re

import torch
from torch.utils.data import Dataset, DataLoader


data_path = './data/SMSSpamCollection'



# 完成数据集类
class MyDataSet(Dataset):

    def __init__(self):
        self.lines = open(data_path, encoding='utf-8').readlines()
        pass

    def __getitem__(self, index):
        # 获取索引对应位置的一条数据
        return self.lines[index]

    def __len__(self):
        # 返回数据的总数量
        return len(self.lines)

if __name__ == '__main__':
    dataset = MyDataSet()
    # print(len(dataset))
    # print(dataset[67])
    data_loader = DataLoader(dataset=dataset, batch_size=2, shuffle=True)
    for i, data in enumerate(data_loader):
        print(i, data)
        break
