import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def tensor_desc():
    """张量的演示"""

    tensor1 = tf.constant(4.0)
    tensor2 = tf.constant([1, 2, 3, 4])
    tensor3 = tf.constant([[1, 2], [2, 3], [4, 9]], dtype=tf.int32)

    print('tensor1:', tensor1)
    print('tensor2:', tensor2)
    print('tensor3:', tensor3)

    return None

def tensor_create():
    """张量的创建"""

    # 固定值张量
    t1 = tf.zeros(shape=[3, 4], dtype=tf.float32, name=None)
    # t2 = tf.zeros_like(t1, dtype=None, name=None)
    t3 = tf.ones(shape=[4, 5], dtype=tf.int32)

    t2 = tf.fill([2, 2], 3, name=None)

    print('tensor1:', t1)
    print('tensor2:', t2)
    print('tensor3:', t3)

    # 随机张量
    # t4 = tf.random_normal_initializer([3, 4], mean=0.5, stddev=1.0)
    # print('tensor4:', t4)

def tensor_retype():
    """Tensor类型修改"""
    tensor = tf.constant([[1, 2], [2, 3], [4, 9]], dtype=tf.int32)
    tensor_new = tf.cast(tensor, dtype=tf.float32)
    print('tensor_old:', tensor)
    print('tensor_new:', tensor_new)

def tensor_reshape():
    """Tensor形状修改"""
    # 占位符
    t_p = tf.placeholder(dtype=tf.float32, shape=[None, None])
    t_p.set_shape([2, 3])
    print('t_p:', t_p)
    a_p = tf.constant([[1, 2], [2, 3], [4, 9]], dtype=tf.int32)
    a_p_new = tf.reshape(a_p, shape=[3, 2])
    print('a_p_new:', a_p_new)



if __name__ == "__main__":
    # tensor_desc()
    # tensor_create()
    # tensor_retype()
    tensor_reshape()



























