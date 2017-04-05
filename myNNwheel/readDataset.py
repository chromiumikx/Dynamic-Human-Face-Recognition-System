#-*- coding: UTF-8 -*-
# 将收集标准数据的模块集成在这里，并在这里更改，不再独立成文件夹，后续将在此文件夹移进下位机代码
import os
import cv2
import numpy as np
import win32api
import win32con
import time

from VK_CODE import VK_CODE

'''
读取文件夹中所有图片成矩阵，并读取用户id
三通道转为一通道
0 ~ 255 转至 -1 ~ +1
'''
def readStandardData(user_names):# 文件夹名字即用户名，为列表传入
    StdFaceMat = []
    StdUserID = []
    for user_name in user_names:
        files_name = os.listdir(user_name + "_data/")
        with open(user_name + "_label.txt", "r+") as fo:
            for file in files_name:
                img = cv2.imread(user_name + "_data/" + file, cv2.IMREAD_GRAYSCALE)# 以灰度模式读取标准数据
                StdFaceMat.append(img / 128 - 1.0)# 读出结果为一list

                # 打开一个文件
                StdUserID.append(fo.read())

    return StdFaceMat, StdUserID

def saveData():
    pass

def collectTest():
    pass

def openFileGetData(Path="data0.txt"):
    pass

##——————————读取训练好的权重矩阵——————————————————
def readWeights(weights_path):
    f=open(weights_path, "r")
    temp=f.readlines()
    weights=[]
    tags=[]
    for i in temp:
        weights.append([float(k) for k in ((i.strip()).split())])
    f.close()
    return weights

if __name__ == "__main__":
    data, _ = readStandardData("ikx")
    pinjie = np.vstack((data[0], data[1]))# 竖向拼接
    cv2.imshow("ME", pinjie)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    data = np.array(data)
    print(data.shape)
    print(pinjie.shape)