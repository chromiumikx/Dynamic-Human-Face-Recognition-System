# -*- coding: UTF-8 -*-
# 将收集标准数据的模块集成在这里，并在这里更改，不再独立成文件夹，后续将在此文件夹移进下位机代码
import os
import cv2
import numpy as np
from myNNwheel.const_config import *

'''
读取文件夹中所有图片成矩阵，并读取用户id
三通道转为一通道
0 ~ 255 转至 -1 ~ +1
返回：图片矩阵和！！用户id
'''


'''
此处修补cv2.imread()函数，
使其输出转换为float类型
'''
def readStandardData(dir_names):  # 文件夹名字即用户名，为列表传入
    StdFaceMat = []
    StdUserID = []
    for user_name in dir_names:
        files_name = os.listdir(user_name + "_data/")
        with open(user_name + "_label.txt", "r+") as fo:
            _id = fo.read()
            for file in files_name:
                '''
                imread返回numpy.uint8大小的数据，无符号，减法操作容易溢出
                故：除以1.0，变成float类型
                '''
                img = ((cv2.imread(user_name + "_data/" + file, cv2.IMREAD_GRAYSCALE))/1.) # 以灰度模式读取标准数据
                img = (img-128)/256.+0.5 # 归一化
                StdFaceMat.append(img) # 读出结果为一list，0 ~ 255 转至 -1 ~ +1
                # 打开一个文件
                StdUserID.append(_id)

    return StdFaceMat, StdUserID


def saveData():
    pass


def collectTest():
    pass


def openFileGetData(Path="data0.txt"):
    pass


##——————————读取训练好的权重矩阵——————————————————
def readWeights(weights_path):
    f = open(weights_path, "r")
    temp = f.readlines()
    weights = []
    # tags = []
    for i in temp:
        weights.append([float(k) for k in ((i.strip()).split())])
    f.close()
    return weights


'''
复制到图片所在的文件夹中运行
'''


def resizePics():
    pic_names = os.listdir()
    pic_names.pop()
    print(pic_names)

    for ipic in pic_names:
        img = cv2.imread(ipic)  # 以灰度模式读取标准数据
        std_iface = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
        if std_iface.ndim == 3:
            gray = cv2.cvtColor(std_iface, cv2.COLOR_BGR2GRAY)
        else:
            gray = std_iface
        cv2.imwrite(ipic, gray)


def my_log_show(iStep, var_name, var_value):
    print("In Step " + str(iStep+1) + " "+var_name+" is:\n", var_value)


'''
以下三个函数为原来的函数，文件被删除，从ConvNN中移回来
'''
def save_face_pics(images, user_name, pic_id = 0):
    _p = os.getcwd()
    os.chdir(_p+"/users_data")
    user_name_add = user_name + "_data"
    # 所指定目录，若不存在则创建
    try:
        os.listdir(os.getcwd()).index(user_name_add)
    except ValueError:
        os.mkdir(user_name_add)

    if images:
        # Done：改造成可以存储多个人脸为多张图片
        i = 0
        for image in images:
            i = i + 1
            if image.ndim == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image  # if语句：如果img维度为3，说明不是灰度图，先转化为灰度图gray，如果不为3，也就是2，原图就是灰度图
            cv2.imwrite(user_name_add + "/face_" + str(pic_id) + "_" + str(i) + ".jpg", gray)

            # 生成一个label.txt标记本文件夹的用户
            # 打开一个文件
            fo = open(user_name + "_label.txt", "w+")
            fo.write("0" + "\n")
            fo.close()

    os.chdir(_p)


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

    faces = face_cascade.detectMultiScale(gray, 1.3, 4)  # 特征的最小、最大检测窗口，它改变检测结果也会改变
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


if __name__ == "__main__":
    data, _ = readStandardData(["ikx"])
    print((np.array(data)).shape)
    pinjie = np.vstack((data[0], data[1]))  # 竖向拼接
    cv2.imshow("ME", pinjie)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    data = np.array(data)
    print(data.shape)
    print(pinjie.shape)
