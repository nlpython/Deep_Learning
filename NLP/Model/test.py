# -*- coding:utf-8 -*-

import jieba
from common.util import preprocess
from CBOW import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

text = '决策树是一类常见的机器学习方法. 以二分类任务为例, 我们希望从给定训练数据集中学得一个模型' \
       '用以对新示例进行分类, 这个把样本分类的任务, 可以看作对"当前样本属于正类吗?"这个问题的"决' \
       '策"或"判定"过程, 顾名思义, 决策树是基于树结构来进行决策的, 这恰是人类在面临决策问题时的' \
       '一种很自然的处理机制. 例如, 我们要对"这是好瓜吗?"这样的问题进行决策时, 通常会进行一系列' \
       '的判断或"子决策": 我们先看"它是什么颜色的?", 如果是"青绿色", 则我们再看"它的根缔是什么' \
       '形态?", 如果是"蜷缩", 我们再判断"它敲起来是什么声音?", 最后, 我们得出最终决策: 这是一' \
       '好瓜.'

text = '较差的氧化性能是钛基合金在高温组织应用中增加使用的主要障碍。为了将这些合金的使用温度提高到550°C(典型的温度极限)以上，需要仔细研究，以了解成分对ti基合金氧化行为的作用[1-3]。为了克服这一限制，ti基合金生产了抗氧化性能显著提高的合金，如β-21S，并开发了涂层和预氧化技术[1,4 - 6]。虽然我们很容易将在一定氧化条件下观察到的有限数量的成分的氧化行为(例如氧化速率定律、氧气进入深度和氧化垢厚度)推断为更广泛的成分范围，在文献中有许多例子，观察到偏离预期关系的情况[7,8]。'
# print(text)
seg_list = jieba.cut(text)
word_list = " ".join(seg_list)
corpus, word_to_id, id_to_word = preprocess(word_list)
# print(corpus)
print(id_to_word)

window_size = 2
hidden_size = 50
batch_size = 5
max_epoch = 200

vocab_size = len(word_to_id)
contexts, target = create_contexts_target(corpus, window_size)



# 生成模型
model = CBOW(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam()

trianer = Trainer(model, optimizer)
trianer.fit(contexts, target, max_epoch, batch_size)
trianer.plot()

# 保存数据
word_vecs = model.word_vecs


fig = plt.figure()
ax = Axes3D(fig)
# ax.scatter(word_vecs[:, 0], word_vecs[:, 1], word_vecs[:, 2])

# plt.show()
# k-meansx训练及可视化
y_pre = KMeans(n_clusters=5, random_state=9).fit_predict(word_vecs)
print(y_pre)
# ax.scatter(word_vecs[:, 0], word_vecs[:, 1], word_vecs[:, 2], c=y_pre)
# plt.show()


mask = np.unique(y_pre)
tmp = []
for v in mask:
    tmp.append(np.sum(y_pre==v))
ts = np.max(tmp)

max_v = mask[np.argmax(tmp)]
print(max_v)

for k in range(5):
       for i in range(len(y_pre)):
              if y_pre[i] == k:
                     print(id_to_word[i], end=' ')
       print()

# pre_score = calinski_harabasz_score(x, y_pre)
# print(pre_score)  # 此值越大越好
# sc = silhouette_score(x, y_pre)
# print(sc)




