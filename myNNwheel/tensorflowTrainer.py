import tensorflow as tf
import numpy as np
from readDataset import readStandardData

import os


image_size = 64
my_saver = None
save_path = "/models/model.ckpt"

def preOperate(img):
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img  # if语句：如果img维度为3，说明不是灰度图，先转化为灰度图gray，如果不为3，也就是2，原图就是灰度图

    return (img-128)/128.

def getPatch(x, y, patch_size):
    step_start = 0
    while step_start < len(x):
        step_end = step_start + patch_size
        if step_end < len(x):
            yield x[step_start:step_end], y[step_start:step_end]
        step_start = step_end


'''
标准数据拼接、加载
最好对图像进行预处理
加载标准输入数据，并进行矩阵拼接，将两张图片拼接在一起作为一个输入
'''
def reconbineDatasets(directories):
    train_pics, train_user_id = readStandardData(directories)

    train_samples = []
    train_labels = []
    two_pics = []
    for i in range(len(train_pics)):
        for j in range(len(train_pics)):
            # 输出 [0,1] 表示是相同
            if train_user_id[i] == train_user_id[j]:
                train_labels.append([0, 1.])
            else:
                # 输出 [1,0] 表示相异
                train_labels.append([1., 0])
            # 拼接 竖向
            two_pics = np.vstack((train_pics[i], train_pics[j]))
            # .resize()是一个操作型函数
            two_pics.resize((1, two_pics.size))
            train_samples.append(two_pics[0])
    return train_samples, train_labels


dire = ["ikx", "sb", "qin"]
train_samples, train_labels = reconbineDatasets(dire)
x = np.array(train_samples)
y = np.array(train_labels)

dire_test = ["qin"]
test_samples, test_labels = reconbineDatasets(dire_test)
x_test = np.array(test_samples)
y_test = np.array(test_labels)


def run_train(x, y, x_test, y_test, image_size, my_saver, save_path):
    '''
    step-1. 定义计算图：
    1. 输入、输出占位符
    2. 变量
    3. 激励函数
    3. 损失熵
    4. 优化算法
    5. 训练步长和初始化
    '''
    graph = tf.Graph()
    with graph.as_default():
        # 占位符，等待传入数据
        x_ph = tf.placeholder(tf.float32, [None, image_size*image_size*2])
        y_ph = tf.placeholder(tf.float32, [None, 2])

        # 变量，将要训练的参数
        '''
        此处，表示出，只有一层隐含层，神经元数目为1024
        '''
        W1 = tf.Variable(tf.zeros([image_size*image_size*2, 1024]))
        b1 = tf.Variable(tf.zeros([1024]))

        W2 = tf.Variable(tf.zeros([1024, 2]))
        b2 = tf.Variable(tf.zeros([2]))

        # 激励函数
        l1 = tf.nn.softmax(tf.matmul(x_ph,W1)+b1)
        l2_output = tf.nn.softmax(tf.matmul(l1, W2) + b2)

        # 损失熵
        cross_entropy  = -tf.reduce_sum(y_ph * tf.log(l2_output))

        # 训练步长，优化器，目的
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(l2_output, 1), tf.argmax(y_ph, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        '''
        保存训练好的模型，在Graph之后保存这个图
        '''
        my_saver = tf.train.Saver()



    with tf.Session(graph=graph) as sess:
        '''
        初始化变量Weights、biases
        '''
        init = tf.global_variables_initializer()

        '''
        step-2. 实例化计算数据流图：
        1. 生产会话
        2. .run(init) # 初始化运行
        '''
        sess.run(init)


        '''
        step-3. 分批次训练：
        1. 载入分批数据
        2. 填充数据
        3. 设定优化器
        4. 启动会话
        '''
        for i in range(20):
            for batch_xs, batch_ys in getPatch(x, y, 100):
                # 传入每次的训练数据，字典形式
                _, accuracy_value = sess.run([train_step, accuracy], feed_dict={x_ph: batch_xs, y_ph: batch_ys})
            print("Accuracy after step"+str(i)+": ", accuracy_value)

        print("Whole accuracy: ", sess.run(accuracy, feed_dict={x_ph: x, y_ph: y}))

        '''
        保存模型
        '''
        if os.path.isdir(save_path):
            _save_path = my_saver.save(sess, save_path)
            print("1:Model Save in: %s"%_save_path)
        else:
            os.makedirs(save_path)
            _save_path = my_saver.save(sess, save_path)
            print("2:Model Save in: %s"%_save_path)


    '''
    step-4. 关闭会话：
    1. 调用close
    2. 也可以使用with...as代码块
    '''
    # sess.close()



def run_inference(x_test, y_test, image_size, my_saver, save_path):
    '''
    step-1. 定义计算图：
    1. 输入、输出占位符
    2. 变量
    3. 激励函数
    3. 损失熵
    4. 优化算法
    5. 训练步长和初始化
    '''
    graph = tf.Graph()
    with graph.as_default():
        # 占位符，等待传入数据
        x_ph = tf.placeholder(tf.float32, [None, image_size*image_size*2])
        y_ph = tf.placeholder(tf.float32, [None, 2])

        # 变量，将要训练的参数
        '''
        此处，表示出，只有一层隐含层，神经元数目为1024
        '''
        W1 = tf.Variable(tf.zeros([image_size*image_size*2, 1024]))
        b1 = tf.Variable(tf.zeros([1024]))

        W2 = tf.Variable(tf.zeros([1024, 2]))
        b2 = tf.Variable(tf.zeros([2]))

        # 激励函数
        l1 = tf.nn.softmax(tf.matmul(x_ph,W1)+b1)
        l2_output = tf.nn.softmax(tf.matmul(l1, W2) + b2)

        correct_prediction = tf.equal(tf.argmax(l2_output, 1), tf.argmax(y_ph, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        my_saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        '''
        初始化变量Weights、biases
        '''
        init = tf.global_variables_initializer()

        '''
        step-2. 实例化计算数据流图：
        1. 生产会话
        2. .run(init) # 初始化运行
        '''
        sess.run(init)
        my_saver.restore(sess, save_path)

        '''
        step-3. 分批次训练：
        1. 载入分批数据
        2. 填充数据
        3. 设定优化器
        4. 启动会话
        '''
        print("Whole accuracy: ", sess.run(accuracy, feed_dict={x_ph: x_test, y_ph: y_test}))


    '''
    step-4. 关闭会话：
    1. 调用close
    2. 也可以使用with...as代码块
    '''
    # sess.close()


if __name__ == "__main__":
    run_train(x,y,x_test,y_test,image_size,my_saver,save_path)
    run_inference(x_test,y_test,image_size,my_saver,save_path)