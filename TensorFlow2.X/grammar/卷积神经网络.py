import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 不显示等级2以下的提示信息
import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
tf.random.set_seed(2345)
import matplotlib.pyplot as plt


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255
    y = tf.cast(y, dtype=tf.int32)
    return x, y

(x, y), (x_val, y_val) = datasets.cifar100.load_data()
y = tf.squeeze(y, axis=1)
y_val = tf.squeeze(y_val, axis=1)
train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.map(preprocess).shuffle(10000).batch(128)
test_db = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_db = test_db.map(preprocess).batch(128)
# sample = next(iter((train_db)))
# print('sample:', sample[0].shape, sample[1].shape)


conv_layers = [     # 5 unit of conv + max pooling
    # unit 1    64个卷积核 两次卷积
    layers.Conv2D(64, kernel_size=[3, 3], strides=1, padding="same", activation='relu'),
    layers.Conv2D(64, kernel_size=[3, 3], strides=1, padding="same", activation='relu'),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),
    # unit 2    128个卷积核 两次卷积
    layers.Conv2D(128, kernel_size=[3, 3], strides=1, padding="same", activation='relu'),
    layers.Conv2D(128, kernel_size=[3, 3], strides=1, padding="same", activation='relu'),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),
    # unit 3    256个卷积核 两次卷积
    layers.Conv2D(256, kernel_size=[3, 3], strides=1, padding="same", activation='relu'),
    layers.Conv2D(256, kernel_size=[3, 3], strides=1, padding="same", activation='relu'),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),
    # unit 4    512个卷积核 两次卷积
    layers.Conv2D(512, kernel_size=[3, 3], strides=1, padding="same", activation='relu'),
    layers.Conv2D(512, kernel_size=[3, 3], strides=1, padding="same", activation='relu'),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),
    # unit 5    512个卷积核 两次卷积
    layers.Conv2D(512, kernel_size=[3, 3], strides=1, padding="same", activation='relu'),
    layers.Conv2D(512, kernel_size=[3, 3], strides=1, padding="same", activation='relu'),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same")
]

def main():
    accuracys = []

    # [b, 32, 32, 3] => [b, 1, 1, 512]
    con_net = Sequential(conv_layers)
    con_net.build(input_shape=[None, 32, 32, 3])
    # x = tf.random.normal([4, 32, 32, 3])
    # out = con_net(x)
    # print(out.shape)

    # [b, 512] => [b, 100]
    fc_net = Sequential([
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(100, activation=None)
    ])
    fc_net.build(input_shape=[None, 512])

    # 优化器
    optimizer = optimizers.Adam(lr=1e-4)
    for epoch in range(500):
        for step, (x, y) in enumerate(train_db):

            with tf.GradientTape() as tape:
                # [b, 32, 32, 3] => [b, 1, 1, 512]
                out = con_net(x)
                # [b, 1, 1, 512] => [b, 512]
                out = tf.reshape(out, shape=[-1, 512])
                # [b, 512] => [b, 100]
                logits = fc_net(out)

                # one-hot
                y_one_hot = tf.one_hot(y, depth=100)
                # compute loss
                loss = tf.losses.categorical_crossentropy(y_one_hot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)

            variables = con_net.trainable_variables + fc_net.trainable_variables    # 列表直接相加
            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))

            if step % 100 == 0:
                print('loss:', float(loss))

        total_num, total_correct = 0, 0
        for step, (x, y) in enumerate(test_db):
            out = con_net(x)
            out = tf.reshape(out, shape=[-1, 512])
            logits = fc_net(out)

            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.reduce_sum(tf.cast(tf.equal(pred, y), dtype=tf.int32))

            total_correct += int(correct)
            total_num += x.shape[0]


        accuracy = total_correct / total_num
        accuracys.append(accuracy)
        print('accuracy:', accuracy)

    plt.plot(np.arange(len(accuracys)), accuracys)
    plt.show()


if __name__ == "__main__":
    main()