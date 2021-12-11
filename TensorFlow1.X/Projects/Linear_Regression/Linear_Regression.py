import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def linear_regression():
    """
    自实现一个线性回归
    :return:
    """
    # 增加命名空间
    with tf.variable_scope('prepare_data'):
        # 1.准备数据
        x = tf.random_normal(shape=[100, 1])
        y_true = tf.matmul(x, [[0.8]]) + 0.7

    with tf.variable_scope('create_model'):
        # 2.定义权重和变量
        weights = tf.Variable(initial_value=tf.random_normal(shape=[1, 1]))
        bias = tf.Variable(initial_value=tf.random_normal(shape=[1, 1]))
        y_predict = tf.matmul(x, weights) + bias

    with tf.variable_scope('loss_function'):
        # 3.构造损失函数
        error = tf.reduce_mean(tf.square(y_predict - y_true))
    with tf.variable_scope('optimizer'):
        # 4.优化损失
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)

    # 显式初始化变量
    init = tf.global_variables_initializer()

    # 收集变量
    tf.summary.scalar('error', error)
    tf.summary.histogram('weights', weights)
    tf.summary.histogram('bias', bias)
    # 合并变量
    merged = tf.summary.merge_all()

    # 开启会话
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init)

        # 创建事件文件
        file_writer = tf.summary.FileWriter("./tmp/linear", graph=sess.graph)\
        #

        # 查看初始化模型参数之后的值
        print(f"初始化模型参数:\n 权重: {weights.eval()}, 偏置: {bias.eval()}, 损失: {error.eval()}")

        # 实例化Saver对象
        saver = tf.train.Saver()

        # 开始训练
        for i in range(1000):
            sess.run(optimizer)
            print(f"训练后模型参数:\n 权重: {weights.eval()}, 偏置: {bias.eval()}, 损失: {error.eval()}")
            # 运行合并变量操作
            summary = sess.run(merged)
            # 将每次迭代后的变量写入事件文件
            file_writer.add_summary(summary, i)

            # 保存模型
            if i % 10 == 0:
                saver.save(sess, './tmp/model/linear_model.ckpt')

        # 加载模型
        # if os.path.exists('./tmp/model/checkpoint'):  # 如果文件存在
        #     saver.restore(sess, './tmp/model/linear_model.ckpt')

        print(f"模型参数:\n 权重: {weights.eval()}, 偏置: {bias.eval()}, 损失: {error.eval()}")


    return None

def load_model():
    # 实例化Saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # 加载模型
        if os.path.exists('tmp/model/checkpoint'):  # 如果文件存在
            saver.restore(sess, './tmp/model/linear_model.ckpt')

        # print(f"模型参数:\n 权重: {weights.eval()}, 偏置: {bias.eval()}, 损失: {error.eval()}")


if __name__ == "__main__":
    linear_regression()
    # load_model()