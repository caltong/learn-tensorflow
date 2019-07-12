from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 查看mnist数据集结构
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print(mnist.train.images.shape, mnist.train.labels.shape)  # images.shape （55000,784) labels.shape (55000,10)
print(mnist.test.images.shape, mnist.test.labels.shape)  # images.shape （10000,784) labels.shape (10000,10)
print(mnist.validation.images.shape, mnist.validation.labels.shape)  # images.shape （5000,784) labels.shape (5000,10)

sess = tf.InteractiveSession()  # 创建session
x = tf.placeholder(tf.float32, [None, 784])  # 创建输入placeholder shape与mnist数据对应
W = tf.Variable(tf.zeros([784, 10]))  # 初始化W为0
b = tf.Variable(tf.zeros([10]))  # 初始化b为0

# 实现softmax regression
y = tf.nn.softmax(tf.add(tf.matmul(x, W)), b)

# 定义交叉熵
y_ = tf.placeholder(tf.float32, [None, 10])  # y' placeholder 定义
cross_entropy = tf.reduce_mean(tf.reduce_sum(y_ * tf.log(y)))  # 交叉熵

# 定义训练优化算法及超参数
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

tf.global_variables_initializer().run()  # 初始化全局变量

# 迭代训练
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # 判断是否正确
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 计算平均正确率
print(accuracy.eval(({x: mnist.test.images, y_: mnist.test.labels})))  # 使用test数据评估正确率
