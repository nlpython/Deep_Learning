import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 不显示等级2以下的提示信息
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import matplotlib.pyplot as plt


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

model = Sequential([
    # [b, 784] => [b, 10]
    layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10)
])

# model.build(input_shape=[None, 784])
# model.summary()


model.compile(optimizer=optimizers.Adam(lr=0.001),
              loss=tf.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
    )
model.fit(train_db, epochs=10,
          validation_data=test_db, validation_freq=2
    )
print('network performance:')
print(x_test.shape)
print(y_test.shape)
model.evaluate(test_db)

# x_var, y_true = next(iter(test_db))
# plt.imshow(x_var.numpy()[2].reshape((28, 28)))
# plt.show()
# print(model.predict_proba(x_var))

