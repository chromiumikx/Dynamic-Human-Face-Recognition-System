import tensorflow as tf
import numpy as np
from readDataset import readStandardData

'''
step-1. 定义计算图：
1. 输入、输出占位符
2. 变量
3. 激励函数
3. 损失熵
4. 优化算法
5. 训练步长和初始化
'''
# 占位符，等待传入数据
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder('float',[None,10])

# 变量，将要训练的参数
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# 激励函数
y = tf.nn.softmax(tf.matmul(x,W)+b)

# 损失熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# 训练步长，优化器，目的
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 初始化变量Weights、biases
init = tf.global_variables_initializer()

'''
step-2. 实例化计算数据流图：
1. 生产会话
2. .run(init) # 初始化运行
'''
# 开始会话
sess = tf.Session()
sess.run(init)


'''
step-3. 分批次训练：
1. 载入分批数据
2. 填充数据
3. 设定优化器
4. 启动会话
'''
# 训练循环
for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)

    # 传入每次的训练数据，字典形式
    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

'''
step-4. 关闭会话：
1. 调用close
2. 也可以使用with...as代码块
'''
sess.close()
