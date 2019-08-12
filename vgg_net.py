from datetime import datetime
import math
import time
import tensorflow as tf

batch_size = 32
num_batches = 100


# 创建卷积层函数并将本层参数存入列表
def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    n_in = input_op.get_shape()[-1].value  # 获取tensor中最后一个数据 为通道数 channel last

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w", shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())  # The TensorFlow contrib module will not be included in TensorFlow 2.0.
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
    # flatten
    shp = pool5.get_shape()  # 获取pool5 shape
    flattened_shape = shp[1].value * shp[2].value * shp[3].value  # 计算flaten后的维度
    reshape1 = tf.reshape(pool5, [-1, flattened_shape], name='resh1')  # 用tf.reshape 转换维度
    # part 6
    fc6 = fc_op(reshape1, 'fc6', n_out=4096, p=p)
    fc6_dropout = tf.nn.dropout(fc6, keep_prob=keep_prob, name='fc6_drop')
    # part 7
    fc7 = fc_op(fc6_dropout, 'fc7', n_out=4096, p=p)
    fc7_dropout = tf.nn.dropout(fc7, keep_prob=keep_prob, name='fc7_dropout')
    # part 8
    fc8 = fc_op(fc7_dropout, 'fc8', n_out=1000, p=p)
    softmax = tf.nn.softmax(fc8)
    predictions = tf.argmax(softmax, 1)
    return predictions, softmax, fc8, p


# 评测函数
def time_tensorflow_run(session, target, feed, info_string):
    num_steps_burn_in = 10  # 预热轮数
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target, feed_dict=feed)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:  # 每10轮输出一次信息
                print('%s: step %d, duration = %.3f' % (datetime.now(), i - num_steps_burn_in, duration))
            total_duration = total_duration + duration
            total_duration_squared = total_duration_squared + duration * duration
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' % (datetime.now(), info_string, num_batches, mn, sd))


def run_benchmark():
    with tf.Graph().as_default():
        # 生成随机数据 标准差0.1 正态分布
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size, image_size, image_size, 3], dtype=tf.float32, stddev=1e-1))

        keep_prob = tf.placeholder(tf.float32)
        preditcions, softmax, fc8, p = inference_op(images, keep_prob)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        time_tensorflow_run(sess, preditcions, {keep_prob: 1.0}, 'Forward')
        objective = tf.nn.l2_loss(fc8)
        grad = tf.gradients(objective, p)
        time_tensorflow_run(sess, grad, {keep_prob: 0.5}, 'Forward-backward')


run_benchmark()
