import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(22)
np.random.seed(22)
assert tf.__version__.startswith('2.')

# the most frequest words
total_words = 10000
max_review_len = 80
embedding_len = 100
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=total_words)
# 将多个句子设置为同一长度的句子
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)

# x_train: [b, 80]  x_test: [b, 80]
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_db = train_db.shuffle(1000).batch(128, drop_remainder=True)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.batch(128, drop_remainder=True)

print('x_train:', x_train.shape, tf.reduce_max(x_train), tf.reduce_min(x_train))
print('y_train:', x_test.shape)

class MyRNN(keras.Model):

    def __init__(self, units):
        super(MyRNN, self).__init__()
        self.state0 = [tf.zeros([128, units]), tf.zeros([128, units])]
        self.state1 = [tf.zeros([128, units]), tf.zeros([128, units])]

        # [b, 80] => [b, 80, 100]
        self.embedding = layers.Embedding(total_words, embedding_len, input_length=max_review_len)
        # [b, 80, 100], h_dim: 64
        # RNN: cell1, cell2, cell3
        # SimpleRNN
        self.rnn_cell0 = layers.LSTMCell(units, dropout=0.2)
        self.rnn_cell1 = layers.LSTMCell(units, dropout=0.2)

        # fc, [b, 80, 100] => [b, 64] => [b, 1]
        self.outlayer = layers.Dense(1)

    def call(self, inputs, training=None):
        """
        net(x) net(x, training=True) :train mode
        net(x, training=False): test
        :param inputs: [b, 80]
        :param training:
        :return:
        """
        # [b, 80]
        x = inputs
        # embedding: [b, 80] => [b, 80, 100]
        x = self.embedding(x)
        # rnn cell compute
        # [b, 80, 100] => [b, 64]
        state0 = self.state0
        state1 = self.state1
        for word in tf.unstack(x, axis=1):   # word: [b, 100]
            # x * wxh + h * whh
            # out0: [b, 64]
            out0, state0 = self.rnn_cell0(word, state0, training)
            # out1: [b, 64]
            out1, state1 = self.rnn_cell1(out0, state1, training)

        # out: [b, 64] => [b, 1]
        x = self.outlayer(out1)
        # p(y is pos|x)
        prob = tf.sigmoid(x)

        return prob

def main():
    units = 64
    epoch = 4

    model = MyRNN(units)
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=tf.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    model.fit(train_db, epochs=epoch, validation_data=test_db)
    model.evaluate(test_db)

if __name__ == '__main__':
    main()
