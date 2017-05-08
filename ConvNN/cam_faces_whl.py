#-*- coding: UTF-8 -*-
# 显示一律使用英文
# CV用于作人脸检测，切图，归一化

import os
import cv2
import tkinter as tk
import numpy as np
from PIL import Image,ImageDraw
from ConvNN.para_config import *
from ConvNN.io_whl import *
from ConvNN.detection_whl import *


def catchUserFace():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

    user_name = input("Input your name:")
    user_id = input("Input your ID:")

    i = 0
    j = 0
    while(True):
        # Capture frame-by-frame
        # frame的宽、长、深为：(480, 640, 3)
        # 后续窗口需要建立和调整，需要frame的大小
        _, frame = cap.read()
        cv2.flip(frame, 1, frame) # mirror the image 翻转图片

        face_area = detect_faces(frame, face_cascade)
        for (x1,y1,x2,y2) in face_area:
            cv2.rectangle(frame,(x1,y1),(x2,y2),(100,0,0),1)

        i = i+1
        if i == 3:
            face_mat = get_faces_mat(frame, face_area)
            # ！！！空列表 [] ，在if语句中 等价于 False或None？？？
            if face_mat:
                j = j + 1
                save_face_pics(face_mat, user_name, j)
                cv2.imshow('Cut Face', face_mat[0]) # getFacesMat()返回值为二维列表，是多个face的数值矩阵

        if (i>=3) and (i<=4):
            for (x1, y1, x2, y2) in face_area:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

        if i > 4:
            i = 0

        cv2.imshow('Face Detect',frame)

        if (cv2.waitKey(1) & 0xFF == ord('q')) or (j == 20): # j：录取照片的数量
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()



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



if __name__ == "__main__":

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
