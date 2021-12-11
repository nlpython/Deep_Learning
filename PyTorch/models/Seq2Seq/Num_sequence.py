
class Num_sequnece:
    PAD_TAG = "PAD"
    PAD = 10
    UNK_TAG = "UNK"
    UNK = 11
    SOS_TAG = 'SOS'
    SOS = 12
    EOS_TAG = 'EOS'
    EOS = 13

    def __init__(self):
        self.dict = {str(i): i for i in range(10)}
        self.dict[self.PAD_TAG] = self.PAD
        self.dict[self.UNK_TAG] = self.UNK
        self.dict[self.SOS_TAG] = self.SOS
        self.dict[self.EOS_TAG] = self.EOS

        self.inverse_dict = {value: key for key, value in self.dict.items()}

    def transform(self, sentence, max_len=9, add_eos=False):
        if len(sentence) > max_len:
            sentence = sentence[: max_len]

        sentence_len = len(sentence)

        if add_eos:
            sentence = sentence + [self.EOS]
        if sentence_len < max_len:
            sentence = sentence + [self.PAD for _ in range(max_len - sentence_len)]

        return [self.dict.get(i, self.UNK) for i in sentence]

    def inverse_transform(self, indices):
        return [self.inverse_dict.get(i, self.UNK_TAG) for i in indices]

    def __len__(self):
        return len(self.dict)


if __name__ == '__main__':
    ns = Num_sequnece()
    print(ns.dict)
    print(ns.inverse_dict)
