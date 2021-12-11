import tensorflow as tf
import numpy as np

def tensorflow_demo():
    """
    TensorFlow的基本结构
    :return:
    """
    # TensorFlow实现加法运算
    a_t = tf.constant(2)
    b_t = tf.constant(3)
    c_t = a_t + b_t
    print(c_t)

    # 开启会话
    with tf.Session() as sess:
        c_t_value = sess.run(c_t)
        print(c_t_value)

def graph_demo():
    # TensorFlow实现加法运算
    a_t = tf.constant(2)
    b_t = tf.constant(3)
    c_t = a_t + b_t
    print(c_t)

    # 自定义图
    new_g = tf.Graph()
    # 在自己的图中定义数据和操作
    with new_g.as_default():
        a_new = tf.constant(20)
        b_new = tf.constant(30)
        c_new = a_new + b_new
        print('c_new.graph:', c_new.graph)
    # 开启会话
    with tf.compat.v1.Session(graph=new_g) as new_sess:
        c_new_value = new_sess.run((c_new))
        print('c_new_value:', c_new_value)
        print('c_new_value图:', new_sess.graph)
        # 写入本地生成events文件
        writer = tf.summary.trace_on(graph=True, profiler=True)  # 开启Trace，可以记录图结构和profile信息
        # 进行训练
        with writer.as_default():
            tf.summary.trace_export(name="model_trace", step=0, profiler_outdir='logs')  # 保存Trace信息到文件

        writer.close()

    return None

if __name__ == "__main__":
    # tensorflow_demo()
    graph_demo()