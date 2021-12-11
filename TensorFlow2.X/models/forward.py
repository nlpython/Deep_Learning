import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 不显示等级2以下的提示信息
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets


# 下载数据集
# x: [60k, 28, 28]
# y: [60k]
(x, y), (x_test, y_test) = datasets.mnist.load_data()

x = tf.convert_to_tensor(x, dtype=tf.float32) / 255
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32) / 255
y = tf.convert_to_tensor(y, dtype=tf.int32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)

train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)
# sample = next(iter(train_db))


# [b, 784] -> [b, 256] -> [b, 128] -> [b, 10]
# [dim_in, dim_out], [dim_out
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

# set learning rate
lr = 0.01


for epoch in range(10):
    # 前向传播算法
    # h1 = x @ w1 + b1
    for step, (x, y) in enumerate(train_db):
        # x: [b, 28, 28] -> [b, 784]
        x = tf.reshape(x, shape=[-1, 784])

        with tf.GradientTape() as tape:
            # [b, 784] @ [784, 256] + [256] => [b, 256]
            h1 = x @ w1 + b1
            h1 = tf.nn.relu(h1)

            # [b, 256] @ [256, 128] + [128] => [b, 128]
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)

            # [b, 128] @ [128, 10] + [10] => [b, 10]
            out = h2 @ w3 + b3


            # out: [b, 10]   y: [b]
            y_one_hot = tf.one_hot(y, depth=10)
            # Then out: [b, 10]   y: [10, b]

            # compute loss
            # mse = mean(sum(y - out)^2)
            # loss = tf.square(y_one_hot - out)
            # loss = tf.reduce_mean(loss)
            loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_one_hot, out, from_logits=True))
            # loss = tf.reduce_mean(tf.losses.MSE(y, out))
            # loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_one_hot - out)))

        # compute gradients
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        # w1' = w1 - lr * grads[0]
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])
        if step % 100 == 0:
            print(step, 'loss:', float(loss))

    # test
    # [w1, b1, w2, b2, w3, b3]
    total_correct, total_num = 0, 0
    for step, (x, y) in enumerate(test_db):
        # [b, 28, 28] -> [b, 784]
        x = tf.reshape(x, shape=[-1, 784])

        # [b, 784] => [b, 10]
        h1 = tf.nn.relu(x @ w1 + b1)
        h2 = tf.nn.relu(h1 @ w2 + b2)
        out = h2 @ w3 + b3

        prob = tf.nn.softmax(out, axis=1)

        pred = tf.argmax(prob, axis=1)
        pred = tf.cast(pred, dtype=tf.int32)

        correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
        correct = tf.reduce_sum(correct)

        total_correct += int(correct)
        total_num += x.shape[0]

    acc = total_correct / total_num
    print('acc:', acc)