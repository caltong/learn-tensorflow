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


# 将数据存到list中
cifar_data = []
for i in range(5):
    filename = 'data_batch_' + str(i + 1)
    cifar_data.append(unpickle(filename))


def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var

