import cv2
import numpy as np
from ConvNN.para_config import *


def detect_faces(image_name, face_cascade):
    if type(image_name) == str:
        img = cv2.imread(image_name)
        print("TTT")
    else:
        img = image_name

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img  # if语句：如果img维度为3，说明不是灰度图，先转化为灰度图gray，如果不为3，也就是2，原图就是灰度图

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # 1.2和5是特征的最小、最大检测窗口，它改变检测结果也会改变
    result = []
    for (x, y, width, height) in faces:
        result.append((x, y, x + width, y + height))  # 返回结果为：人脸区域的左上角和右下角点坐标
    return result  # 返回列表或空列表[]


def get_faces_mat(image_name, face_area): # face_Area是左上角和右下角坐标(x1,y1,x2,y2)
    if face_area:
        if type(image_name) == "str":
            img = cv2.imread(image_name)
        else:
            img = image_name

        result = []
        for (x1,y1,x2,y2) in face_area:
            # 切图索引：第一个是宽，第二个是长
            iface_mat = img[y1:y2, x1:x2]
            # 标准化图像大小
            # 归一化可尝试64*64，可对比各种分辨率的识别率
            std_iface_mat = cv2.resize(iface_mat, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
            result.append(std_iface_mat)
        return result # 返回一个二维数组
