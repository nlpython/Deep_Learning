import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def pict_read(file_list):
    """
    狗图片读取案例
    :return:
    """

    # 1.构建文件名队列
    file_queue = tf.train.string_input_producer(file_list)
    # 2.读取与解码
    reader = tf.WholeFileReader()
    # key是文件名, value是一张图片的原始编码形式
    key, value = reader.read(file_queue)

    # 开始解码
    img = tf.image.decode_jpeg(value)

    # 图像的形状, 类型修改
    image_resized = tf.image.resize_images(img, [200, 200])

    # 静态形状修改
    image_resized.set_shape(shape=[200, 200, 3])

    # 3.批处理
    image_batch = tf.train.batch([image_resized], batch_size=100, num_threads=1, capacity=100)

    # 开启会话
    with tf.Session() as sess:
        # 开启线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)


        key_new, value_new, img_new, image_resized_new, image_batch_new = sess.run([key, value, img, image_resized, image_batch])
        print('key_new:', key_new)
        print('value_new:', value_new)
        print('img_new:', img_new)
        print('image_resized_new:', image_resized_new)
        print('image_batch_new:', image_batch_new)

        # 回收线程
        coord.request_stop()
        coord.join(threads)



    return None

if __name__ == "__main__":
    # 构造路径 + 文件名列表
    file_name = os.listdir('imgs')
    # 拼接路径 + 文件名
    file_list = [os.path.join('imgs', file) for file in file_name]

    pict_read(file_list)