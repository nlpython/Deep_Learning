import nltk
from nltk.corpus import wordnet

# 获取car的同义词簇群
print(wordnet.synsets('car'))

# [Synset('car.n.01'), Synset('car.n.02'), Synset('car.n.03'), Synset('car.n.04'), Synset('cable_car.n.01')]
# 获取对car.n.01簇的描述
car = wordnet.synset('car.n.01')
print(car.definition())

# 获取car.n.01簇中的词
print(car.lemma_names())

# 计算语义相似度[0, 1]
novel = wordnet.synset('novel.n.01')
dog = wordnet.synset('dog.n.01')
motorcycle = wordnet.synset('motorcycle.n.01')
print(car.path_similarity(novel))
print(car.path_similarity(motorcycle))
print(car.path_similarity(dog))
