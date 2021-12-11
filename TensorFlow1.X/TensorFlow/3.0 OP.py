import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def variable_create():
    """变量的创建"""
    a = tf.Variable(initial_value=50)
    b = tf.Variable(initial_value=60)
    c = tf.add(a, b)
    print('a:', a)
    print('b:', b)
    print('c:', c)

    # 初始化变量
    init = tf.global_variables_initializer()

    # with tf.compat.v1.Session() as sess:
    sess = tf.Session()
    # 运行初始化
    sess.run(init)
    a_value, b_value, c_value = sess.run([a, b, c])
    print('a_value:', a_value)
    print('b_value:', b_value)
    print('c_value:', c_value)

    return None

if __name__ == "__main__":
    variable_create()