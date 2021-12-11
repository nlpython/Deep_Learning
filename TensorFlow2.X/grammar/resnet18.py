import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 不显示等级2以下的提示信息
import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from ResNet import resnet18
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

def main():
    accuracys = []

    # [b, 32, 32, 3] => [b, 1, 1, 512]
    model = resnet18()
    model.build(input_shape=(None, 32, 32, 3))

    # 优化器
    optimizer = optimizers.Adam(lr=1e-4)

    for epoch in range(30):
        for step, (x, y) in enumerate(train_db):

            with tf.GradientTape() as tape:
                # [b, 32, 32, 3] => [b, 100]

                logits = model(x)

                # one-hot
                y_one_hot = tf.one_hot(y, depth=100)
                # compute loss
                loss = tf.losses.categorical_crossentropy(y_one_hot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print('loss:', float(loss))

        total_num, total_correct = 0, 0
        for step, (x, y) in enumerate(test_db):
            logits = model(x)

            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.reduce_sum(tf.cast(tf.equal(pred, y), dtype=tf.int32))

            total_correct += int(correct)
            total_num += x.shape[0]


        accuracy = total_correct / total_num
        accuracys.append(accuracy)

        print(epoch, ':', 'accuracy:', accuracy)

    plt.plot(np.arange(len(accuracys)), accuracys)
    plt.show()


if __name__ == "__main__":
    main()