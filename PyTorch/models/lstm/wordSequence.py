

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

        idx = 2
        for word in self.count.keys():
            self.dict[word] = idx
            idx += 1
        # 得到一个反转字典
        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, sentence, max_len=30):
        # idx = 0
        # for key, value in self.dict.items():
        #     print(key, value)
        #     idx += 1
        #     if idx == 100:
        #         break
        # exit()
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

    def __len__(self):
        return len(self.dict)

if __name__ == '__main__':
    pass







