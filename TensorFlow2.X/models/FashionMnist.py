import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 不显示等级2以下的提示信息
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255
    y = tf.cast(y, dtype=tf.int32)
    return x, y

(x, y), (x_test, y_test) = datasets.fashion_mnist.load_data()

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.map(preprocess).shuffle(10000).batch(128)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(128)

model = Sequential([
    # [b, 784] => [b, 10]
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10)
])
model.build(input_shape=[None, 784])
# model.summary()
# w' = w - lr*grads[i]
optimizer = optimizers.Adam(lr = 0.001)

def main():
    """
    fashionmnist数据集
    :return:
    """
    for epcho in range(30):
        for step, (x, y) in enumerate(train_db):
            # x: [b, 28, 28] => [b, 784]
            # y: [b]
            x = tf.reshape(x, [-1, 784])

            with tf.GradientTape() as tape:
                # [b, 784] => [b, 10]
                logits = model(x)
                y_onehot = tf.one_hot(y, depth=10)
                loss1 = tf.reduce_mean(tf.losses.MSE(y_onehot, logits))
                loss2 = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True))

            grads = tape.gradient(loss2, model.trainable_variables)
            # 更新
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print(f"epoch: {epcho}, loss_mse: {loss1}, loss_ce: {loss2}")

        # test
        total_correct, total_num = 0, 0
        for step, (x, y) in enumerate(test_db):
            # x: [b, 28, 28] => [b, 784]
            # y: [b]
            x = tf.reshape(x, [-1, 784])
            logits = model(x)
            # logits => prob
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)
            # pred: [b]  y: [b]
            accuracy = tf.equal(pred, y)
            accuracy = tf.reduce_sum(tf.cast(accuracy, dtype=tf.int32))
            total_correct += int(accuracy)
            total_num += x.shape[0]

        accuracy = total_correct / total_num
        print(f"accuracy: {accuracy}.")



    return None

if __name__ == "__main__":
    main()