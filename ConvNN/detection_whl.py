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

import cv2
from ConvNN.para_config import *


def detect_faces(image_name, face_cascade):
    """检测人脸区域

    Args：
        image_name：待检测的人脸图片矩阵或图片路径
        face_cascade：一个用于检测人脸的分类器对象，为避免调用detect函数时每次重新载入这个分类器对象，故从外部传入

    Inputs：
        载入图片

    Output：
        无

    Returns：
        检测到的人脸区域，是一个二维列表
    """

    if type(image_name) == str:
        img = cv2.imread(image_name)
        print("TTT")
    else:
        img = image_name

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img  # if语句：如果img维度为3，说明不是灰度图，先转化为灰度图gray，如果不为3，也就是2，原图就是灰度图

    faces = face_cascade.detectMultiScale(gray, 1.3, 4)  # 特征的最小、最大检测窗口，它改变检测结果也会改变
    result = []
    for (x, y, width, height) in faces:
        # lock_down_bias增加下巴，去除头发等
        result.append((x, y+lock_down_bias, x + width, y + height+lock_down_bias))  # 返回结果为：人脸区域的左上角和右下角点坐标
    return result  # 返回列表或空列表[]


def get_faces_mat(image_name, face_area): # face_Area是左上角和右下角坐标(x1,y1,x2,y2)
    """获取人脸数据矩阵

    Args：
        image_name：待检测的人脸图片矩阵或图片路径
        face_area：人脸区域坐标

    Inputs：
        载入图片

    Output：
        无

    Returns：
        返回检测到的人脸区域数据，并标准化为image_size*image_size大小的矩阵
    """

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
