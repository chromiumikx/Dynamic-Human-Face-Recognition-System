#-*- coding: UTF-8 -*-
# 显示一律使用英文
# CV用于作人脸检测，切图，归一化

import os
import cv2
import time
import tkinter as tk
import numpy as np
from ConvNN.para_config import *
from ConvNN.io_whl import *
from ConvNN.detection_whl import *
from ConvNN.CNN_whl import *


def catchUserFace():
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
            expand_dataset('E:/Cache/GitHub/Dynamic-Human-Face-Recognition-System/ConvNN/users_data/'+user_name+'_data')
            x, y, x_test, y_test = reconbine_dataset([user_name], [non_user_dir]) # 不能以字符串输入，要以列表形式输入
            train(x, y, x_test, y_test, False, user_name)
            break

    # # When everything done, release the capture
    # cap.release()
    # cv2.destroyAllWindows()



def showDetection():
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



def collect_user_data():
        # GUI
        # tk实现
        # 实时刷新
        window = tk.Tk()
        window.title('my window')
        window.geometry('200x100')

        b_1 = tk.Button(window, text='catchUserFace', command=catchUserFace).pack()
        b_2 = tk.Button(window, text='showDetection', command=showDetection).pack()

        window.mainloop()

        # face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
        # print(type("test_faces_raw.jpg"))
        # face_areas = detectFaces("test_faces_raw.jpg", face_cascade)
        # gray_mat = cv2.imread("test_faces_raw.jpg")
        # for (x1, y1, x2, y2) in face_areas:
        #     cv2.rectangle(gray_mat, (x1, y1), (x2, y2), (0, 0, 0), 1)
        #
        # cv2.imwrite("test_faces_lock.jpg", gray_mat)
        # cv2.imshow('Face Detect', gray_mat)
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    collect_user_data()
