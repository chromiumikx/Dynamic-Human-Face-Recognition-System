# -*- coding: utf-8 -*-
'''
数据输入输出部分集成于此
'''
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_pics_as_mats(dir_names):
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
    step_start = 0
    while step_start < len(x):
        step_end = step_start + patch_size
        if step_end < len(x):
            yield x[step_start:step_end], y[step_start:step_end]
        step_start = step_end


def show_info(fig_name, y_label, y, y_types, dim=1):
    for i in range(len(y)):
        plt.plot(y[i], label=y_types[i])

    plt.legend()  # 展示图例
    plt.xlabel('Steps')  # 给 x 轴添加标签
    plt.ylabel(y_label)  # 给 y 轴添加标签
    plt.title(fig_name)  # 添加图形标题
    plt.show()


def save_face_pics(images, user_name, pic_id = 0):
    _p = os.getcwd()
    os.chdir(_p+"/users_data")
    user_name_a = user_name + "_data"
    # 所指定目录，若不存在则创建
    try:
        os.listdir(os.getcwd()).index(user_name_a)
    except ValueError:
        os.mkdir(user_name_a)

    if images:
        # Done：改造成可以存储多个人脸为多张图片
        i = 0
        for image in images:
            i = i + 1
            if image.ndim == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image  # if语句：如果img维度为3，说明不是灰度图，先转化为灰度图gray，如果不为3，也就是2，原图就是灰度图
            cv2.imwrite(user_name_a + "/face_" + str(pic_id) + "_" + str(i) + ".jpg", gray)

            # 生成一个label.txt标记本文件夹的用户
            # 打开一个文件
            fo = open(user_name + "_label.txt", "w+")
            fo.write("0" + "\n")
            fo.close()

    os.chdir(_p)


def sort_out_non_user_pics():
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
    _datas = os.listdir(data_dir)

    i=0
    for _i_data in _datas:
        i=i+1
        img = cv2.imread(data_dir+"/"+_i_data, cv2.IMREAD_GRAYSCALE)
        img = img*0.6+100
        cv2.imwrite(data_dir + "/expand_face_" + str(i) + ".jpg", np.floor(img))


    # 扩大数据集：增加各级亮度

    # 扩大数据集：增加噪声



if __name__ == "__main__":
    # a, b = load_pics_as_mats(["temp"])
    # print(a[1])
    expand_dataset('E:/Cache/GitHub/Dynamic-Human-Face-Recognition-System/ConvNN/users_data/ikx_data')
