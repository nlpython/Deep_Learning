import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def create_weights(shape):
    return tf.Variable(initial_value=tf.random_normal(shape=shape, stddev=0.01))

def creat_model(x):
    """
    设计卷积神经网络
    :return:
    """
    # 1.第一个卷积大层
    with tf.variable_scope('conv1'):
        # 卷积层
        # 修改x的形状
        input_x = tf.reshape(x, shape=[-1, 28, 28, 1])
        # 定义filter和bias
        conv1_weights = tf.Variable(initial_value=create_weights([5, 5, 1, 32]))
        # conv1_weights = create_weights([5, 5, 1, 32])
        conv1_bias = create_weights([32])
        conv1_x = tf.nn.conv2d(input_x, filter=conv1_weights, strides=[1, 1, 1, 1], padding="SAME") + conv1_bias

        # 激活层
        relu1_x = tf.nn.relu(conv1_x)

        # 池化层
        pool1_x = tf.nn.max_pool(value=relu1_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 2.第二个卷积大层
    with tf.variable_scope('conv2'):
        # 卷积层

        # 定义filter和bias
        conv2_weights = tf.Variable(initial_value=create_weights([5, 5, 32, 64]))
        conv2_bias = create_weights([64])
        conv2_x = tf.nn.conv2d(input=pool1_x, filter=conv2_weights, strides=[1, 1, 1, 1], padding="SAME") + conv2_bias

        # 激活层
        relu2_x = tf.nn.relu(conv2_x)

        # 池化层
        pool2_x = tf.nn.max_pool(value=relu2_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 3.全连接层
    with tf.variable_scope('full_collection'):
        # [None, 7, 7, 64] -> [None, 7*7*64]
        # [None, 7*7*64] * [7*7*64, 10] = [None, 10]

        x_fc = tf.reshape(pool2_x, shape=[-1, 7*7*64])
        weights_fc = create_weights(shape=[7*7*64, 10])
        bias_fc = create_weights(shape=[10])

        y_predict = tf.matmul(x_fc, weights_fc) + bias_fc

    return y_predict

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
    y_predict = creat_model(x)

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
        print('开始训练...')
        # 开始训练
        for i in range(1000):
            _, loss, accuracy_new = sess.run([optimizer, error, accuracy], feed_dict={x: image, y_true: label})
        print(f"训练后损失为: {loss}")
        print(f"准确率为: {accuracy_new}")

    return None

if __name__ == "__main__":
    full_collection()
