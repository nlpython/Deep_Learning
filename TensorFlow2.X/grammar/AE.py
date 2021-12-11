import os
from PIL import Image
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, Sequential, optimizers


tf.random.set_seed(22)
np.random.seed(22)
assert tf.__version__.startswith('2.')

def save_images(imgs, name):
    new_im = Image.new('L', (280, 280))

    index = 0
    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            im = imgs[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))
            index += 1

    new_im.save(name)

h_dim = 20
batchsz = 512
learning_rate = 0.0001

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.
# we don't need label
train_db = tf.data.Dataset.from_tensor_slices(x_train)
train_db = train_db.shuffle(batchsz * 5).batch(batchsz)
test_db = tf.data.Dataset.from_tensor_slices(x_test)
test_db = test_db.shuffle(batchsz * 5).batch(batchsz)

print(x_train.shape, y_train.shape)

class AE(keras.Model):

    def __init__(self):
        super(AE, self).__init__()

        # Encoders
        self.encoder = Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(h_dim)
        ])

        # Decoders
        self.decoder = Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(784)
        ])

    def call(self, inputs, training=None):
        # [b, 28, 28] => [b, 784]
        x = tf.reshape(inputs, shape=[-1, 784])
        # [b, 784] => [b, 10]
        x = self.encoder(x)
        # [b, 10] => [b, 784]
        out = self.decoder(x)

        return out

model = AE()
model.build(input_shape=(None, 784))
optimizer = optimizers.Adam(lr=learning_rate)
# model.summary()

for epoch in range(100):

    for step, x in enumerate(train_db):

        # [b, 28, 28] => [b, 784]
        x = tf.reshape(x, shape=[-1, 784])

        with tf.GradientTape() as tape:
            x_rec_logits = model(x)

            rec_loss = tf.losses.binary_crossentropy(x, x_rec_logits, from_logits=True)
            rec_loss = tf.reduce_mean(rec_loss)

        grads = tape.gradient(rec_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # evaluation
        x = next(iter(test_db))
        logits = model(tf.reshape(x, [-1, 784]))
        x_hat = tf.sigmoid(logits)
        # [b, 784] => [b, 28, 28]
        x_hat = tf.reshape(x_hat, [-1, 28, 28])

        # [b, 28, 28] => [2b, 28, 28]
        x_concat = tf.concat([x, x_hat], axis=0)
        x_concat = x_hat
        x_concat = x_concat.numpy() * 255.
        x_concat = x_concat.astype(np.uint8)
        save_images(x_concat, 'ae_images/rec_epoch_%d.png'%epoch)

    print(epoch, float(rec_loss))




















