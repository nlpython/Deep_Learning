import tensorflow as tf
import pandas as pd
import numpy as np
import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def read_picture():
    """
    读取图片数据
    :return:
    """
    # 1.构造文件名队列
    file_names = glob.glob('./GenPics/*.jpg')
    file_queue = tf.train.string_input_producer(file_names)

    # 2.读取与解码
    reader = tf.WholeFileReader()
    filename, image = reader.read(file_queue)

    # 解码
    decoded = tf.image.decode_jpeg(image)
    # 更新形状和类型
    decoded.set_shape([20, 80, 3])
    image_cast = tf.cast(decoded, tf.float32)

    # 3.批处理
    filename_batch, image_batch = tf.train.batch([filename, image_cast], batch_size=100, num_threads=1, capacity=200)

    return filename_batch, image_batch

def parse_csv():
    """
    解析csv文件, 将文件名与目标值对应
    :return:
    """
    csv_data = pd.read_csv('GenPics/labels.csv', names=['file_name', 'chars'], index_col=['file_name'])
    # print(csv_data)
    # 根据字母生成对应数字 NZPP -> [13, 25, 15, 15]
    labels = []
    for label in csv_data['chars']:
        letter = list(map(lambda x: ord(x) - ord('A'), list(label)))
        labels.append(letter)
    csv_data['labels'] = labels
    # print(csv_data)

    return csv_data

def filename_to_labels(filename, csv_data):
    """
    将一个样本与目标值一一对应
    通过文件名查表
    :param filename:
    :param csv_data:
    :return:
    """
    labels = []
    for file_name in filename:
        file_num = "".join(list(filter(str.isdigit, str(file_name))))
        target = csv_data.loc[int(file_num), 'labels']
        labels.append(target)

    return np.array(labels)

def create_weights(shape):
    return tf.Variable(initial_value=tf.random_normal(shape=shape, stddev=0.01))

def create_models(x):
    """
    构建卷积神经网络
    :param x: [None, 20, 80, 3]
    :return:
    """
    # 1.第一个卷积大层
    with tf.variable_scope('conv1'):
        # 卷积层
        # 定义filter和bias
        conv1_weights = tf.Variable(initial_value=create_weights([5, 5, 3, 32]))
        # conv1_weights = create_weights([5, 5, 1, 32])
        conv1_bias = create_weights([32])
        conv1_x = tf.nn.conv2d(input=x, filter=conv1_weights, strides=[1, 1, 1, 1], padding="SAME") + conv1_bias

        # 激活层
        relu1_x = tf.nn.relu(conv1_x)

        # 池化层
        pool1_x = tf.nn.max_pool(value=relu1_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        # [None, 20, 80, 3] -> [None, 10, 40, 32]

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
        # [None, 10, 40, 32] -> [None, 5, 20, 64]

    # 3.全连接层
    with tf.variable_scope('full_collection'):
        # [None, 5, 20, 64] -> [None, 5*20*64]
        # [None, 5*20*64] * [5*20*64, 4*26] = [None, 4*26]
        x_fc = tf.reshape(pool2_x, shape=[-1, 5*20*64])
        weights_fc = create_weights(shape=[5*20*64, 4*26])
        bias_fc = create_weights(shape=[4*26])

        y_predict = tf.matmul(x_fc, weights_fc) + bias_fc

    return y_predict


if __name__ == "__main__":
    filename, image = read_picture()
    csv_data = parse_csv()

    # 1.准备数据
    x = tf.placeholder(tf.float32, shape=[None, 20, 80, 3])
    y_true = tf.placeholder(tf.float32, shape=[None, 4*26])

    # 2.构造模型
    y_predict = create_models(x)

    # 3.构造损失函数
    loss_list = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_predict)
    loss = tf.reduce_mean(loss_list)

    # 4.优化损失
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    # 5.计算准确率
    # t1 = tf.argmax(tf.reshape(y_predict, shape=[-1, 4, 26]), axis=2)
    # t2 = tf.argmax(tf.reshape(y_true, shape=[-1, 4, 26]), axis=2)
    # equal_list = tf.reduce_all(tf.equal(t1, t2), axis=1)
    # accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    equal_list = tf.reduce_all(
    tf.equal(tf.argmax(tf.reshape(y_predict, shape=[-1, 4, 26]), axis=2),
             tf.argmax(tf.reshape(y_true, shape=[-1, 4, 26]), axis=2)), axis=1)
    accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 初始化变量
    init = tf.global_variables_initializer()

    # 开启会话
    with tf.Session() as sess:
        sess.run(init)
        # 开启线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(1000):
            filename_value, image_value = sess.run([filename, image])
            # print(filename_value)
            # print(image_value)

            labels = filename_to_labels(filename_value, csv_data)
            # 转为ont-hot编码
            labels_value = tf.reshape(tf.one_hot(labels, depth=26), shape=[-1, 4*26]).eval()

            _, error, accuracy_new = sess.run([optimizer, loss, accuracy], feed_dict={x: image_value, y_true:labels_value})
            # if i % 10 == 0:
            print(f"第{i + 1}次训练, 损失为: {error}, 准确率为: {accuracy_new}.")

        # 回收线程
        coord.request_stop()
        coord.join(threads)
