import sys
sys.path.append('./')
from common.util import preprocess, create_co_matrix, most_similar, ppmi


text = ""
with open('./dataset/simple-examples/data/ptb.test.txt') as f:
    line = f.read()
    text += line
print(text.replace('\n', ' ').replace("'", " '").lower())
corpus, word_to_id, id_to_word = preprocess(text.replace('\n', ' ').replace("'", " '").lower())

print('word_to_id["happy"]:', word_to_id['happy'])

C = create_co_matrix(corpus, vocab_size=len(word_to_id), window_size=2)
W = ppmi(C, verbose=True)

# 降维
# truncated SVD (fast!)
from sklearn.utils.extmath import randomized_svd
U, S, V = randomized_svd(W, n_components=100, n_iter=5, random_state=None)

word_vecs = U
print(word_vecs.shape)

querys = ['you', 'year', 'car', 'toyota']
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
