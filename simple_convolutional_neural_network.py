from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # 读取mnist数据
sess = tf.InteractiveSession()  # 创建session


# 初始化权重函数
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    variable = tf.Variable(initial)
    return variable


# 初始化偏置函数
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    variable = tf.Variable(initial)
    return variable

