import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 不显示等级2以下的提示信息
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
import matplotlib.pyplot as plt
import numpy as np

def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
print('x, y range:', x.shape, y.shape)
X, Y = np.meshgrid(x, y)
print('X, Y maps:', X.shape, Y.shape)
Z = himmelblau([X, Y])

fig = plt.figure('himmelblau')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z)
ax.view_init(60, -30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

x = tf.constant([-4.0, 0.0])

for step in range(100):
    with tf.GradientTape() as tape:
        tape.watch([x])
        y = himmelblau(x)

    grad = tape.gradient(y, [x])[0]
    x -= 0.01 * grad

    print(f"step {step}:, x = {x.numpy()}, f(x) = {y.numpy()}.")
