import tensorflow as tf
import numpy as np
import time

max_steps = 3000
batch_size = 128
data_dir = 'cifar-10-batches-py/'


# 读取数据
def unpickle(file):
    import pickle
    with open(data_dir + file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


cifar_data = []
for i in range(5):
    filename = 'data_batch_' + str(i+1)
    cifar_data[i] = unpickle(filename)
