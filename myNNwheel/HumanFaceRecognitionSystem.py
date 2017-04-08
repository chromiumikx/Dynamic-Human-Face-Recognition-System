import tensorflow as tf
import numpy as np
from myNNwheel.readDataset import readStandardData
from  myNNwheel.camFaces import *
import cv2

image_size = 64
my_saver = None
save_path = "/models/model.ckpt"

def reconbineWithStandard(new_pic_mat, user_pics_mat):
    true_labels = []
    two_pic = []
    two_pics = []
    for i in range(len(user_pics_mat)):
        true_labels.append([0, 1.])
        two_pic = np.vstack((new_pic_mat, user_pics_mat[i]))
        cv2.imshow("The pin", two_pic)
        two_pic.resize((1, two_pic.size))
        two_pics.append(two_pic[0])
    return np.array(two_pics), np.array(true_labels)

def preOperate(img):
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img  # if语句：如果img维度为3，说明不是灰度图，先转化为灰度图gray，如果不为3，也就是2，原图就是灰度图

    return (gray-128)/128.

def inference(two_pics_mat, true_labels, pre_saver, save_path):
    graph = tf.Graph()
    with graph.as_default():
        # 占位符，等待传入数据
        x_ph = tf.placeholder(tf.float32, [None, image_size * image_size * 2])
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
        l1 = tf.nn.softmax(tf.matmul(x_ph, W1) + b1)
        l2_output = tf.nn.softmax(tf.matmul(l1, W2) + b2)

        correct_prediction = tf.equal(tf.argmax(l2_output, 1), tf.argmax(y_ph, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        pre_saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        pre_saver.restore(sess, save_path)
        accuracy_value = sess.run(accuracy, feed_dict={x_ph: two_pics_mat, y_ph: true_labels})
        print(accuracy_value)


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

    user_name = input("Input your name:")
    user_id = input("Input your ID:")

    user_pics_mat, _ = readStandardData([user_name])
    i = 0

    '''
    定义计算图，避免调用函数每次重新定义
    '''
    graph = tf.Graph()
    with graph.as_default():
        # 占位符，等待传入数据
        x_ph = tf.placeholder(tf.float32, [None, image_size * image_size * 2])
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
        l1 = tf.nn.softmax(tf.matmul(x_ph, W1) + b1)
        l2_output = tf.nn.softmax(tf.matmul(l1, W2) + b2)

        correct_prediction = tf.equal(tf.argmax(l2_output, 1), tf.argmax(y_ph, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        pre_saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        pre_saver.restore(sess, save_path)

        while(True):
            # Capture frame-by-frame
            # frame的宽、长、深为：(480, 640, 3)
            # 后续窗口需要建立和调整，需要frame的大小
            _, frame = cap.read()
            cv2.flip(frame, 1, frame)  # mirror the image 翻转图片
            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame  # if语句：如果img维度为3，说明不是灰度图，先转化为灰度图gray，如果不为3，也就是2，原图就是灰度图

            face_area = detectFaces(frame, face_cascade)
            for (x1,y1,x2,y2) in face_area:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(100,0,0),1)

            face_mat = getFacesMat(frame, face_area)
            # ！！！空列表 [] ，在if语句中 等价于 False或None？？？
            # getFacesMat 返回列表，故取第一个即可
            accuracy_value = 0
            if face_mat:
                # getFaceMat返回三通道矩阵，故需要变成灰度
                if face_mat[0].ndim == 3:
                    gray_face_mat = preOperate(cv2.cvtColor(face_mat[0], cv2.COLOR_BGR2GRAY))
                    i = i+1
                x_wait_inference, y_true_labels = reconbineWithStandard(gray_face_mat, user_pics_mat)
                l2_output_value, accuracy_value = sess.run([l2_output,accuracy], feed_dict={x_ph: x_wait_inference, y_ph: y_true_labels})
            else:
                print("No Face")

            if accuracy_value>0.7:
                print("Match Person in: ", accuracy_value)
                print(l2_output_value)
                for (x1, y1, x2, y2) in face_area:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                accuracy_value = 0
            else:
                print("Not Match!!!!!!")

            cv2.imshow('Face Detect',frame)

            if (cv2.waitKey(1) & 0xFF == ord('q')):# j：录取照片的数量
                break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
