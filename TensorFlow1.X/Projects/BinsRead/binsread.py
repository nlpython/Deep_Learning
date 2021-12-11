import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Cifar(object):

    def __init__(self):
        # 初始化操作
        self.height = 32
        self.width = 32
        self.channels = 3

        # 字节数
        self.image_bytes = self.height * self.width * self.channels
        self.label_bytes = 1
        self.all_bytes = self.image_bytes + self.label_bytes


    def read_and_decode(self, file_list):
        # 1.构造文件名队列
        file_queue = tf.train.string_input_producer(file_list)

        # 2.读取与解码
        reader = tf.FixedLengthRecordReader(self.all_bytes)
        key, value = reader.read(file_queue)

        # 解码
        decoded = tf.decode_raw(value, tf.uint8)

        # 切片操作
        label = tf.slice(decoded, [0], [self.label_bytes])
        image = tf.slice(decoded, [self.label_bytes], [self.image_bytes])

        # 调整图片形状
        image_reshaped = tf.reshape(image, shape=[3, 32, 32])
        image_reshaped = tf.transpose(image_reshaped, [1, 2, 0])
        print('image_reshape:', image_reshaped)

        # 调整图像类型
        image_cast = tf.cast(image_reshaped, tf.float32)

        # 3.批处理
        label_batch, image_batch = tf.train.batch([label, image_cast], batch_size=100, num_threads=1, capacity=100)

        # 开启会话
        with tf.Session() as sess:
            # 开启线程
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            key_new, value_new, decoded_new, label_new, image_new, label_finally, image_finally = sess.run([key, value, decoded, label, image, label_batch, image_batch])
            print('key_new:', key)
            print('value_new:', value_new)
            print('decoded_new:', decoded_new)
            print('label_finally:', label_finally)
            print('image_finally:', image_finally)

            # 回收线程
            coord.request_stop()
            coord.join(threads)

        return None



if __name__ == "__main__":
    file_name = os.listdir('cifar-10-batches-bin')
    # 构造文件名路径列表
    file_list = [os.path.join('cifar-10-batches-bin', file) for file in file_name if file[-3:] == 'bin']
    # print('file_list:', file_list)

    # 实例化Cifar
    cifar = Cifar()
    cifar.read_and_decode(file_list)
