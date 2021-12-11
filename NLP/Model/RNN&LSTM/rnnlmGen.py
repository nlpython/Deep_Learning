import sys
sys.path.append('../')
import numpy as np
from common.functions import softmax
from rnnlm import Rnnlm
from common.util import preprocess

class RnnlmGen(Rnnlm):
    def generate(self, start_id, skip_ids=None, sample_size=100):
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1)
            score = self.predict(x)
            p = softmax(score.flatten())

            sampled = np.random.choice(len(p), size=1, p=p)
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))

        return word_ids

if __name__ == '__main__':
    # load data
    text = ""
    with open('../dataset/simple-examples/data/ptb.train.txt') as f:
        line = f.read()
        text += line
    # print(text.replace('\n', ' ').replace("'", " '").lower())
    corpus, word_to_id, id_to_word = preprocess(text.replace('\n', ' ').replace("'", " '").lower())

    vocab_size = len(word_to_id)
    corpus_size = len(corpus)

    model = RnnlmGen()

    # 设定start单词和skip单词
    start_word = 'you'
    start_id = word_to_id[start_word]
    skip_words = ['<unk>', '$']
    skip_ids = [word_to_id[w] for w in skip_words]

    # 生成文本
    word_ids = model.generate(start_id, skip_ids)
    txt = ' '.join([id_to_word[i] for i in word_ids])
    txt = txt.replace(' <eos>', '.\n')
    print(txt)





