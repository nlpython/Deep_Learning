import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def command_demo():
    """
    命令行参数演示
    :return:
    """
    # 定义命令行参数
    tf.app.flags.DEFINE_integer('max_step', 100, '训练模型步数')
    tf.app.flags.DEFINE_string('model_dir', 'Unknown', '模型保存的路径+模型名字')
    # 简化变量名
    FLAGS = tf.app.flags.FLAGS

    print('max_step:', FLAGS.max_step)
    print('model_dir:', FLAGS.model_dir)

    return None

def main(argv):
    print(argv)
    return None


if __name__ == "__main__":
    command_demo()
    tf.app.run()