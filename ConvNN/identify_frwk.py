import cv2
import tensorflow as tf

from  ConvNN.cam_faces_whl import *
from ConvNN.para_config import *
from ConvNN.io_whl import *
from ConvNN.detection_whl import *


def preOperate(img):
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img  # if语句：如果img维度为3，说明不是灰度图，先转化为灰度图gray，如果不为3，也就是2，原图就是灰度图

    cv2.imshow("GGGGG", gray)
    return (gray-128)/256.+0.5


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


if __name__ == "__main__":
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    pre_saver = None
    print("Human Face Recognition System v0.9")
    print("Copyright-Mingthic")
    print("\n")

    from ConvNN.cam_faces_whl import *
    from ConvNN.CNN_whl import *
    while True:
        command = input("Do you want?(1.insert user data 2.train my model):")
        if command == "1":
            collect_user_data()
        if command == "2":
            run_CNN()

        if (command =="q") or (command =="Q"):
            print("Quit.\n")
            break


    '''
    检测识别模块
    '''
    print("Human Face Recognition System v0.9")
    user_name = input("Input your name:")
    user_id = input("Input your id:")
    net_save_path = "/models/model_" + user_name + ".ckpt"
    try:
        os.listdir("/models/").index("model_" + user_name + ".ckpt")
        in_user_list = 1
    except ValueError:
        in_user_list = 0

    if (user_name == "Q") or (user_name == "q") or (user_id == "Q") or (user_id == "q") or (in_user_list == 0):
        print("Quit.\n")
    else:
        '''
        定义计算图，避免调用函数每次重新定义
        '''
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

            pre_saver = tf.train.Saver()

            sess = tf.InteractiveSession()
            tf.global_variables_initializer().run()



            pre_saver.restore(sess, net_save_path)
            cap = cv2.VideoCapture(0)
            lock_face_count = 0
            collect_face_count = 0
            while(True):
                # Capture frame-by-frame
                # frame的宽、长、深为：(480, 640, 3)
                # 后续窗口需要建立和调整，需要frame的大小
                _, frame = cap.read()
                cv2.flip(frame, 1, frame)  # mirror the image 翻转图片
                face_area = detect_faces(frame, face_cascade)
                face_mat = get_faces_mat(frame, face_area)
                save_face_pics(face_mat, "temp", collect_face_count)
                # print(collect_face_count)
                # ！！！空列表 [] ，在if语句中 等价于 False或None？？？
                # getFacesMat 返回列表，故取第一个即可
                for (x1,y1,x2,y2) in face_area:
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(100,0,0),1)

                if face_mat and (lock_face_count == 0):
                    lock_face_count = lock_face_count + 1

                if lock_face_count == 1:
                    collect_face_count = collect_face_count + 1

                if collect_face_count == 20:
                    print("Recognizing...")
                    lock_face_count = 0
                    collect_face_count = 0

                    x_pics, y_labels = load_pics_as_mats(["temp"])
                    x_tt = []
                    y_tt = []
                    for i_pic in x_pics:
                        i_pic.resize((1, image_size * image_size))
                        x_tt.append(i_pic[0])  # resize后只取第一行，否则取的是二维数组，维度大小（1，1024）的
                        y_tt.append([0, 1.])
                    x_tt = np.array(x_tt)
                    y_tt = np.array(y_tt)
                    accuracy_value = 0 # 重置
                    [accuracy_value] = sess.run([accuracy], feed_dict={x_ph: x_tt, y_ph: y_tt, keep_prob: 1})

                    print("Accuracy is: "+str(accuracy_value))
                    if accuracy_value>0.9:
                        for (x1, y1, x2, y2) in face_area:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(frame, 'MATCH', (x1+10, y1), font, 2, (0, 255, 0), 2)
                    else:
                        for (x1, y1, x2, y2) in face_area:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(frame, 'XXXXX', (x1+10, y1), font, 2, (0, 0, 255), 2)

                cv2.imshow('Face Detect',frame)

                if (cv2.waitKey(1) & 0xFF == ord('q')):
                    break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
