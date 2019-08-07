import tensorflow as tf
import matplotlib.pyplot as plt
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
def load_train_data():
    cifar_data = []
    for i in range(5):
        filename = 'data_batch_' + str(i + 1)
        cifar_data.append(unpickle(filename))
    return cifar_data


# 读取test_data
def load_test_data():
    test_data = []
    filename = 'test_batch'
    test_data.append(unpickle(filename))
    return test_data


def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var


train_data = load_train_data()  # 读取训练数据
images_train, labels_train = train_data[0][b'data'], train_data[0][b'labels']  # 图片信息与标签信息 暂时只使用batch_0
test_data = load_test_data()  # 读取测试数据
images_test, labels_test = test_data[0][b'data'], test_data[0][b'labels']  # 图片信息与标签信息

images_train = images_train.reshape(10000, 3, 32, 32)  # reshape维度
images_train = np.rollaxis(images_train, 1, 4)  # channel last
images_test = images_test.reshape(10000, 3, 32, 32)  # reshape维度
images_test = np.rollaxis(images_test, 1, 4)  # channel last


# 随机选取训练集batch_size大小的数据
def get_random_train_data(size):
    index = np.array(range(10000))
    index = np.random.choice(index, size)
    random_images = []
    random_labels = []
    for i in index:
        random_images.append(images_train[i])
        random_labels.append(labels_train[i])
    return random_images, random_labels


# 随机选取测试集batch_size大小的数据
def get_random_test_data(size):
    index = np.array(range(10000))
    index = np.random.choice(index, size)
    random_images = []
    random_labels = []
    for i in index:
        random_images.append(images_test[i])
        random_labels.append(labels_test[i])
    return random_images, random_labels


print(images_train.shape)  # (10000,32,32,3)
print(images_test.shape)  # (10000,32,32,3)

# 定义placeholder
image_holder = tf.placeholder(tf.float32, [batch_size, 32, 32, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])

# 定义网络结构 第一层
weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, wl=0.0)  # kernel1权重
kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding='SAME')  # kernel1输出 same padding
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))  # 第一层bias全部为0
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))  # 添加bias
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                       padding='SAME')  # 池化层尺寸3x3 步长2x2 增加数据丰富性 overlapping pooling
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)  # AlexNet 首先使用 local response normalization

# 第二层
weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, wl=0.0)  # kernel1 权重
kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')  # kernel2输出 same padding
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))  # 第二层bias全部为0.1
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)  # 调换pool和lrn顺序 先lrn后pool
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# 第三层
reshape = tf.reshape(pool2, [batch_size, -1])  # flatten
dim = reshape.get_shape()[1].value  # 获取全连接层单元数量
weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, wl=0.004)  # 全连接层权重，加上weight loss L2 Normalization
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

# 第四层
weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, wl=0.004)  # 全连接层权重，加上weight loss L2 Normalization
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

# 第五层
weight5 = variable_with_weight_loss(shape=[192, 10], stddev=1 / 192.0, wl=0.0)
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.add(tf.matmul(local4, weight5), bias5)


# 计算loss
def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)  # 转为int64类型
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                   name='cross_entropy_per_example')  # 计算总交叉熵
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')  # 求平均交叉熵
    tf.add_to_collection('losses', cross_entropy_mean)  # 加入collection
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


loss = loss(logits, label_holder)  # 计算loss
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)  # Adam
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)  # tf.nn.in_top_k函数求出输出结果中top k的准确率，默认使用top 1，输出分数最高的那一类的准确率

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.train.start_queue_runners()  # 开启多线程操作

# 开始训练
for step in range(max_steps):
    start_time = time.time()
    image_batch, label_batch = get_random_train_data(batch_size)
    print(step)
    _, loss_value = sess.run([train_op, loss], feed_dict={image_holder: image_batch, label_holder: label_batch})
    duration = time.time() - start_time

    if step % 10 == 0:
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)

        format_str = ('step %d, loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
        print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))

# 测试
num_examples = 10000
import math

num_iter = int(math.ceil(num_examples / batch_size))
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_batch, label_batch = get_random_test_data(batch_size)
    predictions = sess.run([top_k_op], feed_dict={image_holder: image_batch, label_holder: label_batch})
    true_count = true_count + np.sum(predictions)
    step = step + 1

precision = true_count / total_sample_count
print('precision @ 1 = %.3f' % precision)
