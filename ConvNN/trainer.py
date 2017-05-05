import tensorflow as tf
import numpy as np
from ConvNN.iowhl import *
from ConvNN.para_config import *


def reconbineDatasets(p_user_dir, n_user_dir):
    p_pics, p_user_id = load_pics_as_mats(p_user_dir)
    n_pics, n_user_id = load_pics_as_mats(n_user_dir)

    train_samples = []
    train_labels = []
    test_samples = []
    test_labels = []
    pic_size = image_size*image_size

    for i in p_pics:
        i.resize((1, pic_size))
        train_samples.append(i[0])# resize后只取第一行，否则取的是二维数组，维度大小（1，1024）的
        train_labels.append([0, 1.])

    for ii in range(10):
        test_samples.append(train_samples.pop())
        test_labels.append(train_labels.pop())

    for j in n_pics:
        j.resize((1, pic_size))
        train_samples.append(j[0])
        train_labels.append([1.0, 0])

    for jj in range(10):
        test_samples.append(train_samples.pop())
        test_labels.append(train_labels.pop())

    return np.array(train_samples), np.array(train_labels), np.array(test_samples), np.array(test_labels)


def add_fc_layer(layer_num, X_input, input_scale, layer_depth, active_function=None, keep_prob=1.0):
    with tf.name_scope("fc_layer_"+str(layer_num)):
        with tf.name_scope("paras"):
            Weights = tf.Variable(tf.truncated_normal([input_scale, layer_depth], stddev=0.1))
            biases = tf.Variable(tf.zeros([layer_depth]))
        if active_function==None:
            return tf.matmul(X_input, Weights)+biases
        else:
            return active_function(tf.matmul(X_input, Weights)+biases)


def add_conv_pool_layer(conv_layer_num, X_input, patch_size=5, input_depth=1, conv_depth=32, active_function=None, keep_prob=1.0):
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


def train(x, y, x_test, y_test, is_load=False):
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

        # 首先载入之前的训练结果
        if is_load:
            my_saver.restore(sess, net_save_path)

        print("running....")

        for i in range(max_steps):
            for batch_xs, batch_ys in getPatch(x, y, 39):
                # 传入每次的训练数据，字典形式
                _, acc_training, loss, output_prediction_val = sess.run([train_step, accuracy, cross_entropy, output_prediction],
                                                                        feed_dict={x_ph: batch_xs, y_ph: batch_ys, keep_prob: 0.5})
                print("ACC:"+str(acc_training))
                print("LOSS:"+str(loss))
                print("OUT:" + str(output_prediction_val))

            accuracy_value = sess.run([accuracy], feed_dict={x_ph: x_test, y_ph: y_test, keep_prob: 1})
            print("ACC of test in "+str(i)+str(accuracy_value))

        '''
        保存模型
        '''
        if os.path.isdir(net_save_path):
            _save_path = my_saver.save(sess, net_save_path)
            print("1:Model Save in: %s" % _save_path)
        else:
            os.makedirs(net_save_path)
            _save_path = my_saver.save(sess, net_save_path)
            print("2:Model Save in: %s" % _save_path)


def interfere(x, y):
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

        my_saver.restore(sess, net_save_path)

        # 预测
        accuracy_val, output_prediction_val = sess.run([accuracy, output_prediction], feed_dict={x_ph: x, y_ph: y, keep_prob: 1})
        print("OUT: "+str(output_prediction_val))
        print("ACC: "+str(accuracy_val))


if __name__ == "__main__":
    # x, y, x_test, y_test = reconbineDatasets(["ikx"], ["lei", "qin", "sb"])# 不能以字符串输入，要以列表形式输入
    # train(x, y, x_test, y_test, False)

    x_pics, y_labels = load_pics_as_mats(["ikx"])
    x_tt = []
    y_tt = []
    for i_pic in x_pics:
        i_pic.resize((1, image_size*image_size))
        x_tt.append(i_pic[0])# resize后只取第一行，否则取的是二维数组，维度大小（1，1024）的
        y_tt.append([0, 1.])
    x_tt = np.array(x_tt)
    y_tt = np.array(y_tt)
    interfere(x_tt, y_tt)
