from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 查看mnist数据集结构
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print(mnist.train.images.shape, mnist.train.labels.shape)  # images.shape （55000,784) labels.shape (55000,10)
print(mnist.test.images.shape, mnist.test.labels.shape)  # images.shape （10000,784) labels.shape (10000,10)
print(mnist.validation.images.shape, mnist.validation.labels.shape)  # images.shape （5000,784) labels.shape (5000,10)

sess = tf.InteractiveSession()  # 创建session
x = tf.placeholder(tf.float32, [None, 784])  # 创建输入placeholder shape与mnist数据对应
