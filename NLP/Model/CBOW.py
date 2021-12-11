import sys
sys.path.append('dataset/')
import numpy as np
from common import config
import pickle
from common.trainer import Trainer
from common.optimizer import Adam
from common.util import create_contexts_target, to_cpu, to_gpu, preprocess
from common.layers import Embedding
from common.layers import NegativeSamplingLoss

# config.GPU = True

class CBOW:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V, H = vocab_size, hidden_size

        # 初始化权重
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(V, H).astype('f')

        # 生成层
        self.in_layers = []
        for i in range(2 * window_size):
            layer = Embedding(W_in)
            self.in_layers.append(layer)
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)

        # 保存权重
        layers= self.in_layers + [self.ns_loss]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 将单词的分布式表示设置为成员变量
        self.word_vecs = W_in

    def forward(self, contexts, target):
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])
        # 输入平均化
        h *= 1 / len(self.in_layers)
        loss = self.ns_loss.forward(h, target)
        return loss

    def backward(self, dout=1):
        dout = self.ns_loss.backward(dout)
        dout *= 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None

if __name__ == '__main__':

    window_size = 5
    hidden_size = 100
    batch_size = 100
    max_epoch = 10

    text = ""
    with open('./dataset/simple-examples/data/ptb.test.txt') as f:
        line = f.read()
        text += line
    print(text.replace('\n', ' ').replace("'", " '").lower())
    corpus, word_to_id, id_to_word = preprocess(text.replace('\n', ' ').replace("'", " '").lower())

    vocab_size = len(word_to_id)

    contexts, target = create_contexts_target(corpus, window_size)
    print(corpus)
    print(contexts)
    print(target)

    if config.GPU:
        contexts, target = to_gpu(contexts), to_gpu(target)

    # 生成模型
    model = CBOW(vocab_size, hidden_size, window_size, corpus)
    optimizer = Adam()

    trianer = Trainer(model, optimizer)
    trianer.fit(contexts, target, max_epoch, batch_size)
    trianer.plot()

    # 保存数据
    word_vecs = model.word_vecs
    if config.GPU:
        word_vecs = to_cpu(word_vecs)
    params = {}
    params['word_vecs'] = word_vecs.astype(np.float16)
    params['word_to_id'] = word_to_id
    params['id_to_word'] = id_to_word
    pkl_file = 'cbow_params.pkl'
    with open(pkl_file, 'wb') as f:
        pickle.dump(params, f, -1)

