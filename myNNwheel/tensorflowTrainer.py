import tensorflow as tf
import numpy as np
from myNNwheel.readDataset import *
import os
from myNNwheel.const_config import *
import matplotlib.pyplot as plt


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
    return np.array(train_samples), np.array(train_labels)


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
        with tf.name_scope("inputs"):
            x_ph = tf.placeholder(tf.float32, [None, image_size*2*image_size], name="x_input")
            y_ph = tf.placeholder(tf.float32, [None, 2], name="y_input")

            x_images = tf.reshape(x_ph, [-1,image_size*2,image_size,1], name="x_reshape")

        '''
        整个运算过程
        '''
        with tf.name_scope("Convolution_Layer"):
            h_pool1 = add_conv_pool_layer(1, x_images, 5, 1, 32, tf.nn.relu)
            h_pool2 = add_conv_pool_layer(2, h_pool1, 5, 32, 64, tf.nn.relu)
            h_pool2_flat = tf.reshape(h_pool2, [-1, 8*2*8*64])

        with tf.name_scope("Full_Connect_Layer"):
            l1 = add_fc_layer(1, h_pool2_flat, 8*2*8*64, 1024, tf.nn.relu)
            output_prediction = add_fc_layer(2, l1, 1024, 2, None)

        #TODO：修改为正则化的cost function

        with tf.name_scope("loss"):
            # 损失熵
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_ph, logits=output_prediction))

        with tf.name_scope("train"):
            # 训练步长，优化器，目的
            train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)


        correct_prediction = tf.equal(tf.argmax(output_prediction, 1), tf.argmax(y_ph, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar('accuracy', accuracy)

        # Merge all the summaries and write them out to /train/logs (by default)
        merged = tf.summary.merge_all()

        '''
        保存训练好的模型，在Graph之后保存这个图
        '''
        my_saver = tf.train.Saver()

        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        # 首先载入之前的训练结果
        my_saver.restore(sess, save_path)

        # '''
        # 初始化变量Weights、biases
        # '''
        # init = tf.global_variables_initializer()
        #
        # '''
        # step-2. 实例化计算数据流图：
        # 1. 生产会话
        # 2. .run(init) # 初始化运行
        # '''
        # sess.run(init)

        '''
        保存变量summary的writer
        '''
        train_writer = tf.summary.FileWriter('/train/logs', sess.graph)

        '''
        step-3. 分批次训练：
        1. 载入分批数据
        2. 填充数据
        3. 设定优化器
        4. 启动会话
        '''
        print("running....")

        # 所指定目录，若不存在则创建
        try:
            os.listdir(os.getcwd()).index("ACC.txt")
        except ValueError:
            f_ACC = open("ACC.txt", "a+")

        f_ACC = open("ACC.txt", "a+")

        for i in range(1):
            for batch_xs, batch_ys in getPatch(x, y, 300):
                # 传入每次的训练数据，字典形式
                _ = sess.run([train_step], feed_dict={x_ph: batch_xs, y_ph: batch_ys})

            summary_, accuracy_value, output_value = sess.run([merged, accuracy, output_prediction],
                                                              feed_dict={x_ph: x_test, y_ph: y_test})

            my_log_show(i, "ACC", accuracy_value)
            f_ACC.write(" "+str(accuracy_value)+" ")
            # my_log_show(i, "Output", output_value)
            # my_log_show(i, "y_ph", y_ph_value)
            # my_log_show(i, "w1", w1_value)
            # my_log_show(i, "h_pool1", h_pool1_value)

        f_ACC.close()

        '''
        保存模型
        '''
        if os.path.isdir(save_path):
            _save_path = my_saver.save(sess, save_path)
            print("1:Model Save in: %s" % _save_path)
        else:
            os.makedirs(save_path)
            _save_path = my_saver.save(sess, save_path)
            print("2:Model Save in: %s" % _save_path)


    '''
    step-4. 关闭会话：
    1. 调用close
    2. 也可以使用with...as代码块
    '''
    # sess.close()


def add_fc_layer(layer_num, X, input_scale, layer_depth, active_function=None):
    with tf.name_scope("fc_layer_"+str(layer_num)):
        with tf.name_scope("paras"):
            Weights = tf.Variable(tf.truncated_normal([input_scale, layer_depth], stddev=0.1))
            biases = tf.Variable(tf.zeros([layer_depth]))
        if active_function==None:
            return tf.matmul(X, Weights)+biases
        else:
            return active_function(tf.matmul(X, Weights)+biases)


'''
卷积层
'''
def add_conv_pool_layer(conv_layer_num, X_input, patch_size=5, input_depth=1, conv_depth=32, act=tf.nn.relu):
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    with tf.name_scope("conv_layer_" + str(conv_layer_num)):
        with tf.name_scope("conv_paras"):
            conv_weights = tf.Variable(tf.truncated_normal(shape=[patch_size*2, patch_size, input_depth, conv_depth], stddev=0.1))
            conv_biases = tf.constant(0.1, shape=[conv_depth])
        h = act(conv2d(X_input, conv_weights) + conv_biases)
        return max_pool_2x2(h)


def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def calculate_accuracy(output_predictions, labels):
    std_SAME_labels = []
    for i in range(len(labels)):
        std_SAME_labels.append([0, 1.])
    std_SAME_labels = np.array(std_SAME_labels)

    # SAME准确率
    SAME_labels = np.equal(np.argmax(std_SAME_labels, 1), np.argmax(labels, 1))# 返回label是1的地方 为True
    SAME_prediction = np.equal(np.argmax(std_SAME_labels, 1), np.argmax(output_predictions, 1))# 返回output是1的地方为True
    accuracy_rate = np.sum(np.cast(np.logical_and(SAME_labels, SAME_prediction), np.float32))/(1.0*np.sum(np.cast(SAME_labels, np.float32)))

    # 召回率（错误的被认为正确）
    DIFF_labels = np.equal(np.argmin(std_SAME_labels, 1), np.argmax(labels, 1))# 返回label是0的地方 为True
    callback_rate = np.sum(np.cast(np.logical_and(DIFF_labels, SAME_prediction), np.float32))/(1.0*np.sum(np.cast(DIFF_labels, np.float32)))

    return accuracy_rate, callback_rate


if __name__ == "__main__":
    my_saver = None
    save_path = "/models/model.ckpt"

    dire = ["ikx", "sb", "qin"]
    x, y = reconbineDatasets(dire)

    dire_test = ["ikx", "qin"]
    x_test, y_test = reconbineDatasets(dire_test)
    run_train(x,y,x_test,y_test,image_size,my_saver,save_path)

    f = open("ACC.txt", "r")
    temp = f.readlines()
    ACC = []
    for i in temp:
        ACC.append([float(k) for k in ((i.strip()).split())])
    f.close()

    plt.plot(ACC[0])
    plt.show()
