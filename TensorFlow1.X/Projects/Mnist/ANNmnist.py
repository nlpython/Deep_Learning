import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def full_collection():
    """
    用全连接来对手写数字进行识别
    :return:
    """
    # 1.准备数据
    mnist = input_data.read_data_sets('mnist_data', one_hot=True)
    x = tf.placeholder(dtype=tf.float32, shape=(None, 784))
    y_true = tf.placeholder(dtype=tf.float32, shape=(None, 10))

    # 2.构建模型
    weights = tf.Variable(initial_value=tf.random_normal(shape=[784, 10]))
    bias = tf.Variable(initial_value=tf.random_normal(shape=[10]))
    y_predict = tf.matmul(x, weights) + bias

    # 3.构造损失函数
    error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))

    # 4.优化损失
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)

    # 5.准确率计算
    equal_list = tf.equal(tf.argmax(y_true, axis=1), tf.argmax(y_predict, axis=1))
    accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 初始化变量
    init = tf.global_variables_initializer()

    # 开启会话
    with tf.Session() as sess:
        sess.run(init)
        image, label = mnist.train.next_batch(100)
        print(f"训练前损失为: {sess.run(error, feed_dict={x: image, y_true: label})}")

        # 开始训练
        for i in range(4000):
            _, loss, accuracy_new = sess.run([optimizer, error, accuracy], feed_dict={x: image, y_true: label})
        print(f"训练后损失为: {loss}")
        print(f"准确率为: {accuracy_new}")

    return None

if __name__ == "__main__":
    full_collection()