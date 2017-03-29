#-*- coding: UTF-8 -*-
# 显示一律使用英文
# CV用于作人脸检测，切图，归一化

import os
import cv2
import tkinter as tk
import numpy as np
from PIL import Image,ImageDraw

def detectFaces(image_name):
    if type(image_name) == "str":
        img = cv2.imread(image_name)
    else:
        img = image_name
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img #if语句：如果img维度为3，说明不是灰度图，先转化为灰度图gray，如果不为3，也就是2，原图就是灰度图

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)#1.3和5是特征的最小、最大检测窗口，它改变检测结果也会改变
    result = []
    for (x,y,width,height) in faces:
        result.append((x,y,x+width,y+height))
    return result

def drawFaces(image_name):
    faces = detectFaces(image_name)
    if faces:
        if type(image_name) == "str":
            img = cv2.imread(image_name)
        else:
            img = image_name
        for (x1,y1,x2,y2) in faces:
            cv2.rectangle(img, (x1,y1), (x2,y2), (0, 0, 255), 3)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# 注意图像数据的存储结构：第一个元素是宽，第二个元素是长
# 故 切图时索引要注意
def cutFaces(image_name):
    faces = detectFaces(image_name)
    if faces:
        img = cv2.imread(image_name)
        i = 0
        result = []
        for (x1,y1,x2,y2) in faces:
            i = i+1
            # 切图索引：第一个是宽，第二个是长
            iface = img[y1:y2, x1:x2]
            std_iface = cv2.resize(iface, (80, 80), interpolation=cv2.INTER_CUBIC)
            result.append(std_iface)
        return result

def savePics(images, directory_name = "pics"):
    # 所指定目录，若不存在则创建
    try:
        os.listdir(os.getcwd()).index(directory_name)
    except ValueError:
        os.mkdir(directory_name)

    i = 0
    for image in images:
        i = i + 1
        cv2.imwrite(directory_name+"\\face_" + str(i) + ".jpg", image)

def capShow():
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        # frame的宽、长、深为：(480, 640, 3)
        # 后续窗口需要建立和调整，需要frame的大小
        ret, frame = cap.read()

        # 识别人脸输出坐标
        result = detectFaces(frame)

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

        b = tk.Button(window, text='move', command=capShow).pack()

        window.mainloop()
