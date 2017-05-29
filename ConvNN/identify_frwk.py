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

from ConvNN.detection_whl import *
from ConvNN.CNN_whl import add_fc_layer, add_conv_pool_layer


if __name__ == "__main__":
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    pre_saver = None
    print("Human Face Recognition System v0.9")
    print("\n")

    from ConvNN.cam_faces_whl import *
    from ConvNN.CNN_whl import *
    run_command = {'1': catch_user_face}
    while True:
        command = '1'
        command = input("Do you want?(1.insert user data):")

        try:
            run_command[command]()
        except KeyError:
            print("No This command.\n")

        if (command =="q") or (command =="Q"):
            print("Quit.\n")
            break


    '''
    检测识别模块
    '''
    print("Human Face Recognition System v0.9")
    user_name = "0" # input("Input your name:") # 不再需要输入用户名，可以自动搜索
    user_id = "0" # input("Input your id:") # 不再需要输入用户名，可以自动搜索
    model_name = load_registred_user()
    net_save_path = {}
    for i_model_name in model_name:
        net_save_path[i_model_name] = "/models/model_" + i_model_name + ".ckpt"
    # try:
    #     os.listdir("/models/").index("model_" + user_name + ".ckpt")
    #     in_user_list = 1
    # except ValueError:
    #     in_user_list = 0

    if False:# (user_name == "Q") or (user_name == "q") or (user_id == "Q") or (user_id == "q") or (in_user_list == 0):
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

                # 保证收集到20张待检测人脸，face_mat为空[]时，collect_face_count不加一
                if face_mat and  lock_face_count == 1:
                    collect_face_count = collect_face_count + 1

                if collect_face_count == 20:
                    print("Recognizing...")
                    lock_face_count = 0
                    collect_face_count = 0

                    x_pics, y_labels = load_pics_as_mats(["temp"])
                    # print(len(x_pics))
                    x_tt = []
                    y_tt = []
                    for i_pic in x_pics:
                        i_pic.resize((1, image_size * image_size))
                        x_tt.append(i_pic[0])  # resize后只取第一行，否则取的是二维数组，维度大小（1，1024）的
                        y_tt.append([0, 1.])
                    x_tt = np.array(x_tt)
                    y_tt = np.array(y_tt)
                    accuracy_value = {} # 重置
                    print(accuracy_value)
                    '''
                    要求系统必须已经有一个用户（模型）
                    找出和现有模型匹配度最大的用户的名字
                    '''
                    target_user_name = model_name[0]
                    for i_model_name in model_name:
                        pre_saver.restore(sess, net_save_path[i_model_name])
                        [accuracy_value[i_model_name]] = sess.run([accuracy], feed_dict={x_ph: x_tt, y_ph: y_tt, keep_prob: 1})
                        if accuracy_value[i_model_name] > accuracy_value[target_user_name]:
                            target_user_name = i_model_name

                    print("Accuracy is: ")
                    print(accuracy_value)
                    if accuracy_value[target_user_name] > 0.9:
                        for (x1, y1, x2, y2) in face_area:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            str_match = 'match ' + target_user_name
                            cv2.putText(frame, str_match, (x1, y1), font, 2, (0, 255, 0), 2)
                    else:
                        for (x1, y1, x2, y2) in face_area:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(frame, 'no this user', (x1, y1), font, 2, (0, 0, 255), 2)

                    # 每次重新录取待检测新用户数据前删除已经检测过的数据
                    _p = os.getcwd()
                    _temp_datas_fn = os.listdir(_p + "/users_data/temp_data")
                    for _it in _temp_datas_fn:
                        os.remove(_p+"/users_data/temp_data/"+_it)

                cv2.imshow('Face Detect',frame) # 显示图像和处理后的结果

                if (cv2.waitKey(1) & 0xFF == ord('q')):
                    break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
