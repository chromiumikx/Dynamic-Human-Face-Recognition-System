import tensorflow as tf
import numpy as np
from readDataset import readStandardData


def getPatch(x, y, patch_size):
    step_start = 0
    while step_start < len(x):
        step_end = step_start + patch_size
        if step_end < len(x):
            yield x[step_start:step_end], y[step_start:step_end]
        step_start = step_end


'''
标准数据拼接、加载
'''
# 加载标准输入数据，并进行矩阵拼接，将两张图片拼接在一起作为一个输入
train_pics, train_user_id = readStandardData(["ikx", "qin"])

train_samples = []
train_labels = []
two_pics = []
for i in range(len(train_pics)):
    for j in range(len(train_pics)):
        # 输出 [0,1] 表示是相同
        if train_user_id[i] == train_user_id[j]:
            # 两张图片位置再对调
            train_labels.append([0, 1.])
            train_labels.append([0, 1.])
        else:
            # 输出 [1,0] 表示相异
            train_labels.append([1., 0])
            train_labels.append([1., 0])
        # 拼接 竖向
        two_pics = np.vstack((train_pics[i], train_pics[j]))
        # .resize()是一个操作型函数
        two_pics.resize((1, two_pics.size))
        train_samples.append(two_pics[0])
        two_pics = np.vstack((train_pics[j], train_pics[i]))
        # .resize()是一个操作型函数
        two_pics.resize((1, two_pics.size))
        train_samples.append(two_pics[0])

x = np.array(train_samples)
y = np.array(train_labels)


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
x_ = tf.placeholder(tf.float32, [None, two_pics.size])
y_ = tf.placeholder(tf.float32, [None, 2])

# 变量，将要训练的参数
W1 = tf.Variable(tf.zeros([two_pics.size, 1000]))
b1 = tf.Variable(tf.zeros([1000]))

W2 = tf.Variable(tf.zeros([1000, 2]))
b2 = tf.Variable(tf.zeros([2]))

# 激励函数
l1 = tf.nn.softmax(tf.matmul(x_,W1)+b1)
l2_output = tf.nn.softmax(tf.matmul(l1, W2) + b2)

# 损失熵
cross_entropy  = -tf.reduce_sum(y_ * tf.log(l2_output))

# 训练步长，优化器，目的
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

'''
初始化变量Weights、biases
'''
init = tf.global_variables_initializer()

'''
step-2. 实例化计算数据流图：
1. 生产会话
2. .run(init) # 初始化运行
'''
sess = tf.Session()
sess.run(init)


'''
step-3. 分批次训练：
1. 载入分批数据
2. 填充数据
3. 设定优化器
4. 启动会话
'''
for i in range(50):
    for batch_xs, batch_ys in getPatch(x, y, 400):
        # 传入每次的训练数据，字典形式
        sess.run(train_step, feed_dict={x_:batch_xs, y_:batch_ys})


correct_prediction = tf.equal(tf.argmax(l2_output, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(sess.run(accuracy, feed_dict={x_: x, y_: y}))


'''
step-4. 关闭会话：
1. 调用close
2. 也可以使用with...as代码块
'''
sess.close()
