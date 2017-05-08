# -*- coding: UTF-8 -*-
# 将收集标准数据的模块集成在这里，并在这里更改，不再独立成文件夹，后续将在此文件夹移进下位机代码
import os
import cv2
import numpy as np

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
