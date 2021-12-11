import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 不显示等级2以下的提示信息
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import matplotlib.pyplot as plt

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255
    y = tf.cast(y, dtype=tf.int32)
    return x, y

# [b, 32, 32, 3]
(x, y), (x_val, y_val) = datasets.cifar10.load_data()
y = tf.squeeze(y)   # 消去多余维度, # [b, 1, 10] => [b, 10]
y_val = tf.squeeze(y_val)
y = tf.one_hot(y, depth=10)
y_val = tf.one_hot(y_val, depth=10)

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.map(preprocess).shuffle(10000).batch(128)
test_db = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_db = test_db.map(preprocess).shuffle(10000).batch(128)

class MyDense(layers.Layer):

    def __init__(self, input_dim, output_dim):
        super(MyDense, self).__init__()
        self.kernel = self.add_variable('w', [input_dim, output_dim])
        # self.bias = self.add_variable('b', [inputs_dim, output_dim])

    def call(self, inputs, training=None):
        x = inputs @ self.kernel
        return x

class MyMoudel(keras.Model):

    def __init__(self):
        super(MyMoudel, self).__init__()
        self.fc1 = MyDense(32*32*3, 256)
        self.fc2 = MyDense(256, 128)
        self.fc3 = MyDense(128, 64)
        self.fc4 = MyDense(64, 32)
        self.fc5 = MyDense(32, 10)

    def call(self, inputs, training=None):
        inputs = tf.reshape(inputs, [-1, 32 * 32 * 3])
        x = self.fc1(inputs)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = tf.nn.relu(x)
        x = self.fc3(x)
        x = tf.nn.relu(x)
        x = self.fc4(x)
        x = tf.nn.relu(x)
        x = self.fc5(x)
        return x

network = MyMoudel()
network.compile(optimizer=optimizers.Adam(lr=0.001),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
network.fit(train_db, epochs=5, validation_data=test_db, validation_freq=1)