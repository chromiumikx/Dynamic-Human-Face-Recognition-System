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


import time
from ConvNN.CNN_whl import *
from ConvNN.detection_whl import *


def catch_user_face():
    """录入用户数据并自动训练用户模型
    
    Args：
        无
    
    Inputs：
        要求输入用户名
        要求输入用户ID

    Output：
        保存用户头像和ID为文件
        扩大和保存用户数据集
        训练和保存用户模型
    
    Returns：
        无
    """

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

    user_name = input("Input your name:")
    user_id = input("Input your ID:")

    print("collecting will start after 3 seconds...")
    time.sleep(4)

    count_collect_internal = 0
    count_collect_num = 0
    while(True):
        # Capture frame-by-frame
        # frame的宽、长、深为：(480, 640, 3)
        # 后续窗口需要建立和调整，需要frame的大小
        _, frame = cap.read()
        cv2.flip(frame, 1, frame) # mirror the image 翻转图片

        face_area = detect_faces(frame, face_cascade)
        for (x1,y1,x2,y2) in face_area:
            cv2.rectangle(frame,(x1,y1),(x2,y2),(100,0,0),1)

        count_collect_internal = count_collect_internal+1
        if count_collect_internal == 3:
            face_mat = get_faces_mat(frame, face_area)
            # ！！！空列表 [] ，在if语句中 等价于 False或None？？？
            if face_mat:
                count_collect_num = count_collect_num + 1
                save_face_pics(face_mat, user_name, count_collect_num)
                cv2.imshow('Cut Face', face_mat[0]) # getFacesMat()返回值为二维列表，是多个face的数值矩阵

        if (count_collect_internal>=3) and (count_collect_internal<=4):
            for (x1, y1, x2, y2) in face_area:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

        if count_collect_internal > 4:
            count_collect_internal = 0

        cv2.imshow('Face Detect',frame)

        if (cv2.waitKey(1) & 0xFF == ord('q')) or (count_collect_num == 20): # j：录取照片的数量
            # When everything done, release the capture
            cap.release()
            cv2.destroyAllWindows()

            # 录取完用户数据后，自动增加数据集一次
            _dir = os.getcwd()
            expand_dataset(_dir+'/users_data/'+user_name+'_data')
            x, y, x_test, y_test = recombine_data([user_name], [non_user_dir]) # 不能以字符串输入，要以列表形式输入
            train(x, y, x_test, y_test, False, user_name)
            break

    # # When everything done, release the capture
    # cap.release()
    # cv2.destroyAllWindows()


def show_detection():
    """显示检测到的人脸

    Args：
        无

    Inputs：
        需要读入级联分类器描述文件分类器haarcascade_frontalface_alt.xml

    Output：
        显示当前摄像头捕获的人脸

    Returns：
        无
    """

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

    while(True):
        # Capture frame-by-frame
        # frame的宽、长、深为：(480, 640, 3)
        # 后续窗口需要建立和调整，需要frame的大小
        _, frame = cap.read()
        cv2.flip(frame, 1, frame) # mirror the image

        # 识别人脸输出坐标
        result = detect_faces(frame, face_cascade)

        # 锁定人脸
        if result:
            for (x1,y1,x2,y2) in result:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),3)
        else:
            pass

        cv2.imshow('Face Detect',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    show_detection()
