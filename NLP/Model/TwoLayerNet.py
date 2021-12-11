import sys
sys.path.append('./')
import numpy as np
from common.layers import Affine, Sigmoid, SoftmaxWithLoss
from common.optimizer import SGD
from dataset import spiral
import matplotlib.pyplot as plt


class NetWork:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        # 初始化权重和偏置
        W1 = 0.01 * np.random.randn(I, H)
        b1 = np.zeros(H)
        W2 = 0.01 * np.random.randn(H, O)
        b2 = np.zeros(O)

        # 生成层
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]
        self.loss_layer = SoftmaxWithLoss()

        # 将权重整理到列表中
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):     # 需反向迭代layer
            dout = layer.backward(dout)
        return None

if __name__ == '__main__':
    # 设定超参数
    max_epoch = 300
    batch_size = 30
    hidden_size = 10
    learning_rate = 1.0

    # 读入数据, 生成模型和优化器
    x, t = spiral.load_data()
    print(x.shape, t.shape)
    model = NetWork(input_size=2, hidden_size=hidden_size, output_size=3)
    optimizer = SGD(lr=learning_rate)

    # 学习用的变量
    data_size = len(x)
    max_iters = data_size // batch_size
    total_loss = 0
    loss_count = 0
    loss_list = []

    for epoch in range(max_epoch):
        # 打乱数据
        idx = np.random.permutation(data_size)
        x = x[idx]
        t = t[idx]

        for iters in range(max_iters):
            batch_x = x[iters * batch_size : (iters+1) * batch_size]
            batch_t = t[iters * batch_size : (iters+1) * batch_size]

            # 计算梯度
            loss = model.forward(batch_x, batch_t)
            model.backward()
            optimizer.update(model.params, model.grads)

            total_loss += loss
            loss_count += 1

            # 定期输出学习过程
            if (iters + 1) % 10 == 0:
                avg_loss = total_loss / loss_count
                print(f'|epoch {epoch+1} | iter {iters+1} / {max_iters} | loss: {avg_loss}.')
                loss_list.append(avg_loss)
                total_loss, loss_count = 0, 0

    plt.figure()
    plt.plot(np.arange(len(loss_list)), loss_list)
    plt.show()

    # 境界領域のプロット
    h = 0.001
    x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
    y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    X = np.c_[xx.ravel(), yy.ravel()]
    score = model.predict(X)
    predict_cls = np.argmax(score, axis=1)
    Z = predict_cls.reshape(xx.shape)
    plt.contourf(xx, yy, Z)
    plt.axis('off')

    # データ点のプロット
    x, t = spiral.load_data()
    N = 100
    CLS_NUM = 3
    markers = ['o', 'x', '^']
    for i in range(CLS_NUM):
        plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, marker=markers[i])
    plt.show()




























