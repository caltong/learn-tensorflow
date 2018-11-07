import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# get data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# variables
batch_size = 100
total_steps = 5000
steps_per_test = 500

# build modle
x = tf.placeholder(tf.float32, [None, 784])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# softmax
y = tf.nn.softmax(tf.matmul(x, w) + b)
# loss cross entropy
y_label = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_mean(y_label * tf.log(y), reduction_indices=[1]))

# train use gradient descent optimizer
train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_label, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# run
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # train 10000 steps
    for step in range(total_steps + 1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train, feed_dict={x: batch_x, y_label: batch_y})
        # test every 100 steps
        if step % steps_per_test == 0:
            print(step, sess.run(accuracy, feed_dict={x: mnist.test.images, y_label: mnist.test.labels}))