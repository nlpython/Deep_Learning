"""
准备数据集
"""
import torch
import numpy as np
import lib
from torch.utils.data import DataLoader, Dataset


class NumDataset(Dataset):

    def __init__(self):
        super(NumDataset, self).__init__()
        # 使用numpy随机创建一个数
        np.random.seed(8)
        self.data = np.random.randint(1, 1e8, size=[100000000])

    def __getitem__(self, idx):
        input = list(str(self.data[idx]))    # ['1', '8', '0', '6', '2', '9', '7']
        label = input + ['0']
        input_length = len(input)
        label_length = len(label)
        return input, label, input_length, label_length

    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    # batch: [(input, label, input_length, label_length), (input, label, input_length, label_length)...]

    batch = sorted(batch, key=lambda x: x[3], reverse=True)   # 根据
    content, label, content_len, label_len = zip(*batch)
    # ret: ([input, input,,,], [label, label,,,])
    content = torch.LongTensor([lib.ns.transform(i, max_len=lib.max_len) for i in content])
    label = torch.LongTensor([lib.ns.transform(i, max_len=lib.max_len+1, add_eos=True) for i in label])
    content_len = torch.LongTensor(content_len)
    label_len = torch.LongTensor(label_len)
    return content, label, content_len, label_len


def get_data_loader(is_train=True):
    num_dataset = NumDataset()
    return DataLoader(num_dataset, batch_size=lib.batch_size, shuffle=is_train, drop_last=True, collate_fn=collate_fn)

if __name__ == '__main__':
    # num_dataset = NumDataset()
    for input, label, len1, len2 in get_data_loader():
        print(input)
        print(label)
        print(len1)
        break