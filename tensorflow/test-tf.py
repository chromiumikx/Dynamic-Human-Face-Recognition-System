import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 占位符，等待传入数据
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder('float',[None,10])

# 变量，将要训练的网络
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# 激励函数
y = tf.nn.softmax(tf.matmul(x,W)+b)

# 损失熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# 训练步长，优化器，目的
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 初始化
init = tf.global_variables_initializer()

# 开始会话
sess = tf.Session()
sess.run(init)

# 训练循环
for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)

    # 传入每次的训练数据，字典形式
    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
