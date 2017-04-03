#-*- coding: UTF-8 -*-
# 显示一律使用英文
# CV用于作人脸检测，切图，归一化

import os
import cv2
import tkinter as tk
import numpy as np
from PIL import Image,ImageDraw

def detectFaces(image_name, face_cascade):
    if type(image_name) == "str":
        img = cv2.imread(image_name)
    else:
        img = image_name

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img #if语句：如果img维度为3，说明不是灰度图，先转化为灰度图gray，如果不为3，也就是2，原图就是灰度图

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)#1.2和5是特征的最小、最大检测窗口，它改变检测结果也会改变
    result = []
    for (x,y,width,height) in faces:
        result.append((x,y,x+width,y+height))
    return result


# 注意图像数据的存储结构：第一个元素是宽，第二个元素是长
# 故 切图时索引要注意
def getFaces(image_name, face_area):# face_Area是左上角和右下角坐标(x1,y1,x2,y2)
    if face_area:
        if type(image_name) == "str":
            img = cv2.imread(image_name)
        else:
            img = image_name

        result = []
        for (x1,y1,x2,y2) in face_area:
            # 切图索引：第一个是宽，第二个是长
            iface = img[y1:y2, x1:x2]
            # 标准化图像大小
            # 归一化可尝试64*64，可对比各种分辨率的识别率
            std_iface = cv2.resize(iface, (64, 64), interpolation=cv2.INTER_CUBIC)
            result.append(std_iface)
        return result


def saveFacePics(images, directory_name, pic_id = 0):
    directory_name = directory_name + "_data"
    # 所指定目录，若不存在则创建
    try:
        os.listdir(os.getcwd()).index(directory_name)
    except ValueError:
        os.mkdir(directory_name)

    if images:
        for image in images:
            if image.ndim == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image  # if语句：如果img维度为3，说明不是灰度图，先转化为灰度图gray，如果不为3，也就是2，原图就是灰度图
            cv2.imwrite(directory_name+"\\face_" + str(pic_id)+".jpg", gray)



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
        cv2.flip(frame, 1, frame)  # mirror the image

        face_area = detectFaces(frame, face_cascade)
        for (x1,y1,x2,y2) in face_area:
            cv2.rectangle(frame,(x1,y1),(x2,y2),(100,0,0),1)

        i = i+1
        if i == 6:
            face = getFaces(frame, face_area)
            if face:
                j = j + 1
                saveFacePics(face, user_name, j)
                cv2.imshow('Cut Face', face[0])# getFaces()返回值为四维，是多个face的数值矩阵

        if (i>6) and (i<8):
            for (x1, y1, x2, y2) in face_area:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

        if i > 9:
            i = 0

        cv2.imshow('Face Detect',frame)

        if (cv2.waitKey(1) & 0xFF == ord('q')) or (j == 20):# j：录取照片的数量
            break

    # 生成一个label.txt标记本文件夹的用户
    # 打开一个文件
    fo = open(user_name+"_label.txt", "w+")
    fo.write(user_id+"\n")
    fo.close()

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
        cv2.flip(frame, 1, frame)  # mirror the image

        # 识别人脸输出坐标
        result = detectFaces(frame, face_cascade)

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
        window.geometry('600x200')

        b = tk.Button(window, text='move', command=catchUserFace).pack()

        window.mainloop()
