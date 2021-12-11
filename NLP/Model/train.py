import sys
sys.path.append('./')
from common.optimizer import SGD
from dataset import spiral
from TwoLayerNet import NetWork
from common.trainer import Trainer

# 设定超参数
max_epoch = 500
batch_size = 30
hidden_size = 10
learning_rate = 1.0

x, t = spiral.load_data()
model = NetWork(input_size=2, hidden_size=hidden_size, output_size=3)
optimizer = SGD(lr=learning_rate)

trainer = Trainer(model=model, optimizer=optimizer)
trainer.fit(x, t, max_epoch, batch_size, eval_interval=10)
trainer.plot()