import fileinput
import torch
from torch.utils.tensorboard import SummaryWriter

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


writer = SummaryWriter()

embedded = torch.randn(100, 50)

meta = list(map(lambda x: x.strip(), fileinput.FileInput('./data/vocab100.csv')))
writer.add_embedding(embedded, metadata=meta)
writer.close()