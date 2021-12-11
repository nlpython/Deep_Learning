import torch
from torch.utils.data import DataLoader, Dataset
import os
import re
import torch.nn as nn

def tokenlize(content):
    filters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>', '\?',
               '@'
        , '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”', '“', "'"]
    text = re.sub('<.*?>', " ", content)
    content = re.sub('|'.join(filters), ' ', content)
    tokens = [i.strip().lower() for i in content.split()]
    return tokens

class ImdbDataset(Dataset):
    def __init__(self, train=True):
        self.train_data_path = '../data/aclImdb/train'
        self.test_data_path = '../data/aclImdb/test'
        data_path = self.train_data_path if train else self.test_data_path

        # 把所有的文件名放入列表
        temp_data_path = [os.path.join(data_path, 'pos'), os.path.join(data_path, 'neg')]
        self.total_file_path = []    # 所有评论文件的路径
        for path in temp_data_path:
            file_name_list = os.listdir(path)
            file_path_list = [os.path.join(path, i) for i in file_name_list if i.endswith('.txt')]
            self.total_file_path.extend(file_path_list)

    def __getitem__(self, idx):
        file_path = self.total_file_path[idx]
        # 获取label
        label_str = file_path.split('\\')[-2]
        label = 0 if label_str == 'neg' else 1
        # 获取内容
        tokens = tokenlize(open(file_path, encoding='utf-8').read())
        return tokens, label

    def __len__(self):
        return len(self.total_file_path)

class Word2Sequence:
    # 特殊字符
    UNK_TAG = 'UNK'
    PAD_TAG = 'PAD'
    UNK = 0
    PAD = 1

    def __init__(self):
        self.dict = {
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD
        }
        self.inverse_dict = {
            self.UNK: self.UNK_TAG,
            self.PAD: self.PAD_TAG
        }
        self.count = {}

    def fit(self, sentence):
        """
        :param sentence: [word1, word2, word3,..]
        :return:
        """
        for word in sentence:
            if word not in self.count.keys():
                self.count[word] = 1
                self.dict[word] = len(self.dict)
                self.inverse_dict[len(self.inverse_dict)] = word
            else:
                self.count[word] += 1

    def build_vocab(self, min=5, max=None):
        """
        生成词典
        :param min: 最少出现次数
        :param max: 最大出现次数
        :param max_features: 一共保留多少词语
        :return:
        """
        # 删除count中小于min的词
        self.count = {word: value for word, value in self.count.items() if value > min}

        max_features = len(self.count)

        for word in self.count.keys():
            self.dict[word] = len(self.dict)
        # 得到一个反转字典
        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, sentence, max_len=30):
        id_list = []
        for word in sentence:
            id_list.append(self.dict.get(word, self.UNK))

        # 对句子进行填充或裁剪
        if len(id_list) > max_len:
            id_list = id_list[:max_len]
        else:
            id_list.extend([self.PAD for _ in range(max_len - len(id_list))])
        return id_list

    def inverse_transform(self, indices):
        word_list = []
        for idx in indices:
            word_list.append(self.inverse_dict.get(idx, self.UNK_TAG))
        return word_list



def collate_fn(batch):
    """
    :param batch: 一个getitem的结果 [[token, label], [token, label]..]
    :return:
    """
    ret = zip(*batch)
    return ret


def get_data_loader(is_train=True):
    imdb_dataset = ImdbDataset(is_train)
    data_loader = DataLoader(imdb_dataset, batch_size=3, shuffle=True, collate_fn=collate_fn)
    return data_loader


# def create_dict(data_loader):
#     word2id = {UNK_TAG: 0, PAD_TAG: 1}
#     id2word = {0: UNK_TAG, 1: PAD_TAG}
#     idx = 2
#     for _, (input, target) in enumerate(data_loader):
#         for seq in input:
#             for word in seq:
#                 if word not in word2id.keys():
#                     word2id[word] = idx
#                     id2word[idx] = word
#                     idx += 1
#     return word2id, id2word

if __name__ == '__main__':
    data_loader = get_data_loader(True)
    ws = Word2Sequence()
    for idx, (input, target) in enumerate(data_loader):
        ws.fit(input[0])
        print(input[0])
        print(ws.transform(input[0]))
        print(ws.inverse_transform(ws.transform(input[0])))
        break





















