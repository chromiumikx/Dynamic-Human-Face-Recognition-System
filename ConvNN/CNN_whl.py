#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2017 Mingthic

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import tensorflow as tf
from ConvNN.io_whl import *
from ConvNN.para_config import *


def recombine_data(p_user_dir, n_user_dir):
    """将数据整理为训练数据和测试数据

    Args：
        p_user_dir：用户数据文件夹
        n_user_dir：非用户数据文件夹

    Inputs：
        要求载入用户和非同户数据

    Output：
        train_samples：用于训练的例子数据
        train_labels：用于训练的标签
        test_samples：用于测试的例子数据
        test_labels：用于测试的标签

    Returns：
        即输出
    """

    p_pics, p_user_id = load_pics_as_mats(p_user_dir)
    n_pics, n_user_id = load_pics_as_mats(n_user_dir)

    all_datas = []
    all_labels = []
    train_samples = []
    train_labels = []
    test_samples = []
    test_labels = []
    pic_size = image_size*image_size

    for i in p_pics:
        i.resize((1, pic_size))
        all_datas.append(i[0]) # resize后只取第一行，否则取的是二维数组，维度大小（1，1024）的
        all_labels.append([0, 1.])

    for j in n_pics:
        j.resize((1, pic_size))
        all_datas.append(j[0])
        all_labels.append([1.0, 0])

    while(len(all_datas)>2):
        train_samples.append(all_datas.pop())
        train_labels.append(all_labels.pop())
        test_samples.append(all_datas.pop())
        test_labels.append(all_labels.pop())

    return np.array(train_samples), np.array(train_labels), np.array(test_samples), np.array(test_labels)


def build_graph():
    """
    关于graph和session、saver等类的关系还需要学习清楚

    """

    graph = tf.Graph()
    with graph.as_default():
        # 占位符，等待传入数据
        with tf.name_scope("inputs"):
            x_ph = tf.placeholder(tf.float32, [None, image_size*image_size], name="x_input")
            y_ph = tf.placeholder(tf.float32, [None, 2], name="y_input")

            x_images = tf.reshape(x_ph, [-1,image_size,image_size,1], name="x_reshape")

        with tf.name_scope("Convolution_Layer"):
            h_pool1 = add_conv_pool_layer(1, x_images, 5, 1, 32, tf.nn.relu)
            h_pool1_dropout = tf.nn.dropout(h_pool1, keep_prob=1)
            h_pool2 = add_conv_pool_layer(2, h_pool1_dropout, 5, 32, 64, tf.nn.relu)
            h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])

        keep_prob = tf.placeholder(tf.float32)
        with tf.name_scope("Full_Connect_Layer"):
            l1 = add_fc_layer(1, h_pool2_flat, 8*8*64, 1024, tf.nn.relu)
            l1_dropout = tf.nn.dropout(l1, keep_prob)
            output_prediction = add_fc_layer(2, l1_dropout, 1024, 2, None)

        with tf.name_scope("loss"):
            # 损失熵
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_ph, logits=output_prediction))

        with tf.name_scope("train"):
            # 训练步长，优化器，目的
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)


        correct_prediction = tf.equal(tf.argmax(output_prediction, 1), tf.argmax(y_ph, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        '''
        保存训练好的模型，在Graph之后保存这个图
        '''
        my_saver = tf.train.Saver()

    return graph, accuracy, output_prediction, my_saver


def add_fc_layer(layer_num, X_input, input_scale, layer_depth, active_function=None, keep_prob=1.0):
    """增加全连接层

    Args：
        layer_num：本全连接层的序号
        X_input：本层的输入
        input_scale：每个输入的维度
        layer_depth：本层的深度，即神经元个数，亦即权重矩阵的列数
        active_function：激活函数
        keep_prob：dropout技术，神经元随机置零时不置零的概率，即留下神经元的概率

    Inputs：
        无

    Output：
        无

    Returns：
        本层的神经元，即经过与权重相乘，加上偏置，再经过激活函数后的输出值
    """

    with tf.name_scope("fc_layer_"+str(layer_num)):
        with tf.name_scope("paras"):
            Weights = tf.Variable(tf.truncated_normal([input_scale, layer_depth], stddev=0.1))
            biases = tf.Variable(tf.zeros([layer_depth]))
        if active_function==None:
            return tf.matmul(X_input, Weights)+biases
        else:
            return active_function(tf.matmul(X_input, Weights)+biases)


def add_conv_pool_layer(conv_layer_num, X_input, patch_size=5, input_depth=1, conv_depth=32, active_function=None, keep_prob=1.0):
    """增加全连接层

    Args：
        conv_layer_num：本卷积层的序号
        X_input：本层的输入
        patch_size：卷积核的大小
        input_depth：每组输入的深度，此处用的灰度图故为1，如果是RGB彩色图像则为3
        conv_depth：本层的深度，即卷积核的个数（对每个输入而言）
        active_function：激活函数
        keep_prob：dropout技术，神经元随机置零时不置零的概率，即留下神经元的概率

    Inputs：
        无

    Output：
        无

    Returns：
        本层的神经元，经过激活、卷积、池化后的输出值
    """

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    with tf.name_scope("conv_layer_" + str(conv_layer_num)):
        with tf.name_scope("conv_paras"):
            conv_weights = tf.Variable(tf.truncated_normal(shape=[patch_size, patch_size, input_depth, conv_depth], stddev=0.1))
            conv_biases = tf.constant(0.1, shape=[conv_depth])
        h = active_function(conv2d(X_input, conv_weights) + conv_biases)
        return max_pool_2x2(h)


def train(x, y, x_test, y_test, is_load=False, user_name=None):
    """训练用户模型

    Args：
        x：用于训练的例子数据
        y：用于训练的标签
        x_test：用于测试的例子数据
        y_test：用于测试的标签
        is_load：训练时是否使用之前训练过的的模型，此处为False，表示每次都重新训练
        user_name：已有数据的用户名字，用于保存模型命名

    Inputs：
        要求载入用户和非同户数据

    Output：
        保存和显示训练过程中的损失函数值和准确度等数据
        保存训练好的模型
        保存已注册用户的名字

    Returns：
        无
    """

    graph = tf.Graph()
    with graph.as_default():
        # 占位符，等待传入数据
        with tf.name_scope("inputs"):
            x_ph = tf.placeholder(tf.float32, [None, image_size*image_size], name="x_input")
            y_ph = tf.placeholder(tf.float32, [None, 2], name="y_input")

            x_images = tf.reshape(x_ph, [-1,image_size,image_size,1], name="x_reshape")

        with tf.name_scope("Convolution_Layer"):
            h_pool1 = add_conv_pool_layer(1, x_images, 5, 1, 32, tf.nn.relu)
            h_pool1_dropout = tf.nn.dropout(h_pool1, keep_prob=1)
            h_pool2 = add_conv_pool_layer(2, h_pool1_dropout, 5, 32, 64, tf.nn.relu)
            h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])

        keep_prob = tf.placeholder(tf.float32)
        with tf.name_scope("Full_Connect_Layer"):
            l1 = add_fc_layer(1, h_pool2_flat, 8*8*64, 1024, tf.nn.relu)
            l1_dropout = tf.nn.dropout(l1, keep_prob)
            output_prediction = add_fc_layer(2, l1_dropout, 1024, 2, None)

        with tf.name_scope("loss"):
            # 损失熵
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_ph, logits=output_prediction))

        with tf.name_scope("train"):
            # 训练步长，优化器，目的
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)


        correct_prediction = tf.equal(tf.argmax(output_prediction, 1), tf.argmax(y_ph, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        '''
        保存训练好的模型，在Graph之后保存这个图
        '''
        my_saver = tf.train.Saver()

        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        net_save_path = "/models/model_" + user_name + ".ckpt"
        # 首先载入之前的训练结果
        if is_load:
            my_saver.restore(sess, net_save_path)

        print("running....")

        log_loss = []
        log_tset_acc = []
        pre_accuracy_test_value = 0

        for i in range(max_steps):
            for batch_xs, batch_ys in get_patch(x, y, 500):
                # 传入每次的训练数据，字典形式
                _, acc_training, train_loss, output_prediction_val = sess.run([train_step, accuracy, cross_entropy, output_prediction],
                                                                        feed_dict={x_ph: batch_xs, y_ph: batch_ys, keep_prob: 0.5})
                print("ACC:"+str(acc_training))
                print("LOSS:"+str(train_loss))
            log_loss.append(train_loss)
            # print("OUT:" + str(output_prediction_val))

            # feed测试集的时候，keep_prob为1
            # 对于训练集，则是0.5或别的，这主要是为了让训练的网络有泛化的能力
            [accuracy_test_value] = sess.run([accuracy], feed_dict={x_ph: x_test, y_ph: y_test, keep_prob: 1})
            print("ACC of test in "+str(i)+": "+str(accuracy_test_value))
            log_tset_acc.append(accuracy_test_value)

            # 测试集和训练集的训练精度达到一定要求即停止训练
            if pre_accuracy_test_value != accuracy_test_value:
                pre_accuracy_test_value = accuracy_test_value
            else:
                break

            if (acc_training>target_accuracy) and (accuracy_test_value>target_accuracy):
                print("Accuracy done.")
                break

        show_info("Training Loss", "LOSS", [log_loss], ["NtoN"])
        show_info("Test Accuracy", "ACC", [log_tset_acc], ["NtoN"])


        '''
        保存模型
        '''
        if os.path.isdir(net_save_path):
            _save_path = my_saver.save(sess, net_save_path)
            print("Added Model Save in: %s" % _save_path)
        else:
            os.makedirs(net_save_path)
            _save_path = my_saver.save(sess, net_save_path)
            print("New Model Save in: %s" % _save_path)

        # 保存已注册的用户记录
        with open("registered_users_name.txt", "a+") as f:
            f.write(" "+user_name)

        print("MODEL TRAINING DONE.")


def interfere(x, y, user_name):
    """载入用户模型进行推理预测

    Args：
        x：用于预测的例子数据
        y：用于预测的标签
        user_name：已有模型的用户名字，用于载入训练好的模型

    Inputs：
        要求载入用户模型

    Output：
        打印预测结果
        打印预测的准确率
        显示隐藏层的信息

    Returns：
        无
    """

    graph = tf.Graph()
    with graph.as_default():
        # 占位符，等待传入数据
        with tf.name_scope("inputs"):
            x_ph = tf.placeholder(tf.float32, [None, image_size*image_size], name="x_input")
            y_ph = tf.placeholder(tf.float32, [None, 2], name="y_input")

            x_images = tf.reshape(x_ph, [-1,image_size,image_size,1], name="x_reshape")

        with tf.name_scope("Convolution_Layer"):
            h_pool1 = add_conv_pool_layer(1, x_images, 5, 1, 32, tf.nn.relu)
            h_pool1_dropout = tf.nn.dropout(h_pool1, keep_prob=1)
            h_pool2 = add_conv_pool_layer(2, h_pool1_dropout, 5, 32, 64, tf.nn.relu)
            h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])

        keep_prob = tf.placeholder(tf.float32)
        with tf.name_scope("Full_Connect_Layer"):
            l1 = add_fc_layer(1, h_pool2_flat, 8*8*64, 1024, tf.nn.relu)
            l1_dropout = tf.nn.dropout(l1, keep_prob)
            output_prediction = add_fc_layer(2, l1_dropout, 1024, 2, None)

        correct_prediction = tf.equal(tf.argmax(output_prediction, 1), tf.argmax(y_ph, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        '''
        保存训练好的模型，在Graph之后保存这个图
        '''
        my_saver = tf.train.Saver()

        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        net_save_path = "/models/model_" + user_name + ".ckpt"
        my_saver.restore(sess, net_save_path)

        # 预测
        accuracy_val, output_prediction_val = sess.run([accuracy, output_prediction],
                                                       feed_dict={x_ph: x, y_ph: y, keep_prob: 1})
        print("OUT: "+str(output_prediction_val))
        print("ACC: "+str(accuracy_val))

        # 显示卷积层的情况
        [h_pool1_val] = sess.run([h_pool1], feed_dict={x_ph: x, y_ph: y, keep_prob: 1})
        show_conv_layers(h_pool1_val)


def run_CNN():
    """
    结合上述两个函数，供识别模块调用

    """

    while True:
        target = input("Train or Interfere? (T/I):")
        if (target=="q") or (target=="Q"):
            break

        if (target == "T") or (target == "t"):
            target = "Train"
            user_name = input("Input your name:")
            # 增加整理的另外一个数据集non_2
            # 作为非用户人脸的参照系，主要由手动挑选的正面人脸组成
            # 目的是为了保持，正的用户人脸集和负的非用户人脸集，在倾斜度、光照、姿态、发型、背景性质（未考虑，可能有影响）等保持一致，只有人脸部分不一致
            # 以避免神经网络学习到错误的性质
            x, y, x_test, y_test = recombine_data([user_name], [non_user_dir]) # 不能以字符串输入，要以列表形式输入
            train(x, y, x_test, y_test, False, user_name)

        if (target == "I") or (target == "i"):
            target = "Interfere"
            model_user_name = input("Input User Model name:") # 决定要使用的用户模型
            user_name = input("Input test name:") # 决定要加载的用户数据T
            x_pics, y_labels = load_pics_as_mats([user_name])
            x_tt = []
            y_tt = []
            for i_pic in x_pics:
                i_pic.resize((1, image_size*image_size))
                x_tt.append(i_pic[0]) # resize后只取第一行，否则取的是二维数组，维度大小（1，1024）的
                y_tt.append([0, 1.])
            x_tt = np.array(x_tt)
            y_tt = np.array(y_tt)
            interfere(x_tt, y_tt, model_user_name)

        print(target+" Done.")
    print("Exit.")


if __name__ == "__main__":
    run_CNN()