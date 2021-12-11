import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 不显示等级2以下的提示信息
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255
    y = tf.cast(y, dtype=tf.int32)
    return x, y

(x, y), (x_test, y_test) = datasets.mnist.load_data()

y = tf.one_hot(y, depth=10)
y_test = tf.one_hot(y_test, depth=10)

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.map(preprocess).shuffle(10000).batch(128)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(128)

def create_save_model():

    model = Sequential([
        # [b, 784] => [b, 10]
        layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(10)
    ])


    model.compile(optimizer=optimizers.Adam(lr=0.001),
                  loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy']
        )
    model.fit(train_db, epochs=3,
              validation_data=test_db, validation_freq=2
        )
    model.evaluate(test_db)

    # 模型的保存与加载

    # 保存权值
    # model.save_weights('weights.ckpt')
    # print('saved weights')
    # 保存模型的所有状态
    model.save('model.h5')
    print('saved total model.')

    # 删除内存中的模型
    del model


def load_model():
    # 重新从本地加载
    network = Sequential([
        # [b, 784] => [b, 10]
        layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(10)
    ])

    network.compile(optimizer=optimizers.Adam(lr=0.001),
                  loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy']
        )
    network.load_weights('weights.ckpt')
    print('load model')
    network.evaluate(test_db)

def load_all():
    print('load model from file')
    network = tf.keras.models.load_model('model.h5')
    network.evaluate(test_db)


if __name__ == "__main__":
    # create_save_model()
    # load_model()
    load_all()
