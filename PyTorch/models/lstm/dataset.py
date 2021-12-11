from torch.utils.data import DataLoader, Dataset
import os
import re
from wordSequence import Word2Sequence
# import pickle
from lib import ws, batch_size


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
        file_path = self.total_file_path[idx]       # ../data/aclImdb/train\pos\5661_10.txt
        # 获取label
        label_str = file_path.split('_')[-1]    # 10.txt

        label = int(label_str.split('.')[0]) - 1    # 0 - 9
        # 获取内容
        tokens = tokenlize(open(file_path, encoding='utf-8').read())
        return tokens, label

    def __len__(self):
        return len(self.total_file_path)

def collate_fn(batch):
    """
    :param batch: 一个getitem的结果 [[token, label], [token, label]..]
    :return:
    """
    import torch
    ret, label = zip(*batch)
    # print('ret', ret)
    content = [ws.transform(i, max_len=20) for i in ret]
    # print(content)
    content = torch.LongTensor(content)
    label = torch.LongTensor(label)
    return content, label


def get_data_loader(is_train=True):
    imdb_dataset = ImdbDataset(is_train)
    data_loader = DataLoader(imdb_dataset, batch_size=batch_size, shuffle=is_train, collate_fn=collate_fn, drop_last=True)
    return data_loader

if __name__ == '__main__':
    import os
    from dataset import tokenlize
    from dataset import ImdbDataset
    import pickle
    from tqdm import tqdm

    dataset = get_data_loader()
    for input, target in dataset:
        break


    # # 保存ws
    # ws = Word2Sequence()
    # path = '../data/aclImdb/train'
    # temp_data_path = [os.path.join(path, 'pos'), os.path.join(path, 'neg')]
    # for data_path in temp_data_path:
    #     file_paths = [os.path.join(data_path, file_name) for file_name in os.listdir(data_path) if file_name.endswith('.txt')]
    #     for file_path in tqdm(file_paths):
    #         sentence = tokenlize(open(file_path, encoding='utf-8').read())
    #         ws.fit(sentence)
    #         # print(ws.dict)
    #
    # ws.build_vocab(min=10)
    #
    # pickle.dump(ws, open('../lstm/ws.pkl', 'wb'))


