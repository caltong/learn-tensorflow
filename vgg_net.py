from datetime import datetime
import math
import time
import tensorflow as tf


# 创建卷积层函数并将本层参数存入列表
def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    n_in = input_op.get_shape()[-1].value  # 获取tensor中最后一个数据 为通道数 channel last

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w", shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        bias_init_value = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_value, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        p = p + [kernel, biases]

        return activation


# 全连接层创建函数
def fc_op(input_op, name, n_out, p):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + 'w', shape=[n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b')
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)  # 可用分别计算后使用tf.nn.relu
        p = p + [kernel, biases]

        return activation


# 最大池化创建函数
def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op, ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1], padding='SAME', name=name)


def inference_op(input_op, keep_prob):
    p = []  # 初始化参数列表
    # part 1
    conv1_1 = conv_op(input_op, 'conv1_1', kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    conv1_2 = conv_op(conv1_1, 'conv1_2', kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    pool1 = mpool_op(conv1_2, 'pool1', kh=2, kw=2, dh=2, dw=2)  # 输出 112x112x64
    # part 2
    conv2_1 = conv_op(pool1, 'conv2_1', kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    conv2_2 = conv_op(conv2_1, 'conv2_2', kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    pool2 = mpool_op(conv2_2, 'pool2', kh=2, kw=2, dh=2, dw=2)  # 输出 56x56x128
    # part 3
    conv3_1 = conv_op(pool2, 'conv3_1', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_2 = conv_op(conv3_1, 'conv3_2', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    pool3 = mpool_op(conv3_2, 'pool3', kh=2, kw=2, dh=2, dw=2)  # 输出 28x28x256
    # part 4
    conv4_1 = conv_op(pool3, 'conv4_1', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_2 = conv_op(conv4_1, 'conv4_1', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool4 = mpool_op(conv4_2, 'conv4_2', kh=2, kw=2, dh=2, dw=2)  # 输出14x14x512
    # part 5
    conv5_1 = conv_op(pool4, 'conv5_1', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_2 = conv_op(conv5_1, 'conv5_2', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool5 = mpool_op(conv5_2, 'pool5', kh=2, kw=2, dh=2, dw=2)  # 输出7x7x512

    shp = pool5.get_shape()  # 获取pool5 shape
    flattened_shape = shp[1].value * shp[2].value * shp[3].value  # 计算flaten后的维度
    reshp = tf.reshape(pool5, [-1, flattened_shape], name='resh1')  # 用tf.reshape 转换维度
    
