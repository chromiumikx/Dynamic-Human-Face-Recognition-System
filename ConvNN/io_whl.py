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

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ConvNN.para_config import *


def load_pics_as_mats(dir_names):
    """载入文件夹中的图片作为矩阵

    Args：
        dir_names：文件夹路径

    Inputs：
        载入图片

    Output：
        无

    Returns：
        归一化过后的人脸数据和ID
    """

    StdFaceMat = []
    StdUserID = []
    _p = os.getcwd()
    os.chdir(_p+"/users_data")
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

    os.chdir(_p)

    return StdFaceMat, StdUserID


def get_patch(x, y, patch_size):
    """取数据集中的一部分

    Args：
        x：输入的人脸数据
        y：输入的标签数据
        patch_size：要取的数据批的大小

    Inputs：
        无

    Output：
        无

    Yields：
        本批次数据x_this_patch，y_this_patch
    """

    step_start = 0
    while step_start < len(x):
        step_end = step_start + patch_size
        if step_end < len(x):
            yield x[step_start:step_end], y[step_start:step_end]
        step_start = step_end


def show_info(fig_name, y_label, y, y_types, dim=1):
    """对plt的封装，显示信息

    Args：
        fig_name：图的名字
        y_label：数据的标签（类型）
        y：要画图的数据
        y_types：用于对比
        dim：数据维度，目前只考虑一元

    Inputs：
        无

    Output：
        显示数据图

    Returns：
        无    
    """

    for i in range(len(y)):
        plt.plot(y[i], label=y_types[i])

    plt.legend()  # 展示图例
    plt.xlabel('Steps')  # 给 x 轴添加标签
    plt.ylabel(y_label)  # 给 y 轴添加标签
    plt.title(fig_name)  # 添加图形标题
    plt.show()


def show_conv_layers(h_pool1_val):
    """以二维形式显示卷积层

    Args：
        h_pool1_val：卷积层的数据

    Inputs：
        无

    Output：
        显示卷积层的数据图像

    Returns：
        无    
    """

    conv_depth = h_pool1_val.shape[3]
    is_show_for_every_batch = 1 # h_pool1_val.shape[0]
    for jj in range(is_show_for_every_batch):
        k = 0
        _t_hstk = ()
        for i in range(int(conv_depth/8)):
            t_k = ()
            for j in range(8):
                _val = h_pool1_val[jj, :, :, k]
                _re_val = cv2.resize(_val, (80, 80), interpolation=cv2.INTER_CUBIC)
                t_k = t_k + (_re_val,)
                k = k + 1
            _hstk_8 = np.hstack(t_k)
            print(_hstk_8.shape)
            _t_hstk = _t_hstk + (_hstk_8,)
        vhstk = np.vstack(_t_hstk)

        cv2.imshow("IMG", vhstk)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def save_face_pics(images, user_name, pic_id = 0):
    """将人脸区域的图片保存下来

    Args：
        images：人脸数据，三维列表，即支持单个图片中多人脸检测后分别保存
        user_name：用于图片命名
        pic_id：便于程序控制命名不冲突

    Inputs：
        无

    Output：
        保存用户人脸图片
        保存用户的标签（ID）

    Returns：
        无    
    """

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


def sort_out_non_user_pics():
    """整理lfw数据集的

    Args：
        无

    Inputs：
        载入数据集的所有图片

    Output：
        检测人脸后切出人脸保存

    Returns：
        无    
    """

    import ConvNN.detection_whl
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    source_folder = "E:/Learn/GraduationProject/DataSets/lfw/" # 源文件目录
    target_folder = "E:/Cache/GitHub/Dynamic-Human-Face-Recognition-System/ConvNN/users_data/non_user_data" # 将要更改成的工作目录

    # 读取所有的图片
    non_users = os.listdir(source_folder)
    i = 0
    for non_user in non_users:
        this_non_user_pics = os.listdir(source_folder+non_user+"/")
        for this_non_user_pic in this_non_user_pics:
            i = i + 1
            this_pic_path = source_folder+non_user+"/" + this_non_user_pic
            img = cv2.imread(this_pic_path)
            face_area = ConvNN.detection_whl.detect_faces(img, face_cascade)
            face_mat = ConvNN.detection_whl.get_faces_mat(img, face_area)
            save_face_pics(face_mat, "non_user", i)
            print("Saved pic "+str(i))


def expand_dataset(data_dir):
    """为录入的用户数据增加干扰和修正

    Args：
        data_dir：待修正扩大数据集的数据路径

    Inputs：
        载入图片

    Output：
        增加亮度后保存
        增加噪声后保存

    Returns：
        无    
    """

    def change_light(delta_light):
        def chglight(x):
            r = x+delta_light
            if r<0:
                r = 0
            elif r>255:
                r = 255

            return r
        return np.frompyfunc(chglight, 1, 1)

    _datas = os.listdir(data_dir)

    i=len(_datas)
    str_ = '_'
    for _i_data in _datas:
        img = cv2.imread(data_dir+"/"+_i_data, cv2.IMREAD_GRAYSCALE)

        # 扩大数据集：增加各级亮度
        for delta_light in [-50, -40, -35, -30, -20, -15 -10, 5, 15, 20, 25, 30, 35, 40]:
            img_1 = change_light(delta_light)(img)
            cv2.imwrite(str_.join((data_dir+'/expand_light', str(i), 'add_light', str(delta_light), '.jpg')),
                        np.floor(img_1.astype(np.float64)))

        i = i + 1

    # 距离特性问题：通过切割部分人脸再resize，得到不同距离下可能的人脸图片
    _datas = os.listdir(data_dir)
    for _i_data in _datas:
        img = cv2.imread(data_dir + "/" + _i_data, cv2.IMREAD_GRAYSCALE)
        # 距离特性问题：通过切割部分人脸再resize，得到不同距离下可能的人脸图片
        for delta_d in [1,2]:
            img_2 = img[delta_d:-delta_d, delta_d:-delta_d]
            img_3 = img[(delta_d+delta_d):, delta_d:-delta_d]
            std_img_2 = cv2.resize(img_2, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
            std_img_3 = cv2.resize(img_3, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(str_.join((data_dir+'/expand_distance', str(i), 'chgdst', str(delta_d), '.jpg')), std_img_2)
            cv2.imwrite(str_.join((data_dir+'/expand_distance', str(i), 'chgdst', str(delta_d), '_other_cut.jpg')), std_img_3)

        i = i + 1

    # 作直方图均衡
    # equ = cv2.equalizeHist(img)


def load_registred_user():
    """载入已注册用户的用户名

    Args：
        无

    Inputs：
        无

    Output：
        无

    Returns：
        列表形式返回用户名 
    """

    _dir = "registered_users_name.txt"
    users = {}
    with open(_dir, "r") as f:
        temps = f.readlines()
        for temp in temps:
            users.update({k: k for k in temp.split()})
        return users


def calc_pic_attributes(pic_dir, user_name):
    _pics = os.listdir(pic_dir)

    # 计算直方图，也为图像亮度属性提供数据
    hists = np.zeros([256, 1])
    for _pic in _pics:
        img = cv2.imread((pic_dir+'/'+_pic),  cv2.IMREAD_GRAYSCALE)
        h = cv2.calcHist([img],[0],None,[256],[0,256])
        hists = hists + h

    average_hist = hists/len(_pics)
    # plt.plot(average_hist, label=user_name)
    # plt.show()

    light = 0
    for i in range(len(average_hist)):
        light = light + i*average_hist[i]

    return light/sum(average_hist)


if __name__ == "__main__":
    # a, b = load_pics_as_mats(["temp"])
    # print(a[1])

    # this_dir = 'E:/Cache/GitHub/Dynamic-Human-Face-Recognition-System/ConvNN/'

    # expand_dataset(this_dir+'users_data/new_non_3_data')

    # users = load_registred_user()
    # print(users)

    print(calc_pic_attributes('users_data/ikx_data', 'ikx'))
    # print(calc_pic_attributes('users_data/ikx1_data', 'ikx1'))
    # print(calc_pic_attributes('users_data/qin_data', 'qin'))
    # print(calc_pic_attributes('users_data/new_non_3_data', 'new_non_3'))
    # print(calc_pic_attributes('users_data/temp_data', 'temp'))
    # plt.legend()  # 展示图例
    # plt.show()
