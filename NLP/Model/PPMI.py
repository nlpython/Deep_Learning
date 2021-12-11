import sys

import matplotlib.pyplot as plt

sys.path.append('./')
import numpy as np
from common.util import preprocess, cos_similarity, create_co_matrix, ppmi

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

C = create_co_matrix(corpus, len(word_to_id))
W = ppmi(C)

np.set_printoptions(precision=3)
print('covariance matrix:', C[0])
print('PPMI:', W[0])

# SVD降维
U, S, V = np.linalg.svd(W)
print('SVD:', U[0])

for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))
plt.scatter(U[:,0], U[:1], alpha=0.5)
plt.show()
