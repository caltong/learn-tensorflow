from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)  # 读取mnist数据集
sess = tf.InteractiveSession()  # 创建默认Interactive Session 后面就不需要指定session运行

in_units = 784  # 输入层
h1_units = 300  # 中间层

# 权重初始化
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32, [None, 784])  # 定义输入x placeholder
keep_prob = tf.placeholder(tf.float32)  # dropout 概率 训练时小于1 预测时等于1

# 定义模型结构
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)

# 定义损失函数 交叉熵
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=1))

# 定义优化方法及超参数
train_step = tf.train.AdadeltaOptimizer(0.3).minimize(cross_entropy)

tf.global_variables_initializer().run()  # 初始化全局变量
for i in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)  # 获取batch数据
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})  # 用batch数据训练

# 评估
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 平均准确率
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))  # 用test数据评估
