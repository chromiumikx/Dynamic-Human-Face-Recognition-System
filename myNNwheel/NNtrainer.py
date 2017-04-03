#-*- coding: UTF-8 -*-
import cv2
import numpy as np
from readDataset import readStandardData
from readDataset import readWeights

def trainNeuralNetwork():
    #_________________数据与代码同一目录时用下列代码_________________________

    np.random.seed(1)

    # ***************数据拼接*******************
    inputs = []
    out_puts = []
    # 加载标准输入数据，并进行矩阵拼接，将两张图片拼接在一起作为一个输入
    input_data_list_1, _ = readStandardData("ikx")
    input_data_list_2, _ = readStandardData("qin")
    for i in range(1000):
        # 0，1随机
        k = np.random.randint(0, 2)

        if i < 200:
            # 输出是 0 时，表示是相同
            out_puts.append(0.0001)

            # 拼接 竖向
            input_data = np.vstack((input_data_list_1[np.random.randint(0, len(input_data_list_1))], input_data_list_1[np.random.randint(0, len(input_data_list_1))]))
            # .resize()是一个操作型函数
            input_data.resize((1, input_data.size))
            inputs.append(input_data[0])
        else:
            out_puts.append(0.9999)

            # 拼接 竖向
            input_data = np.vstack((input_data_list_1[np.random.randint(0, len(input_data_list_1))], input_data_list_2[np.random.randint(0, len(input_data_list_2))]))
            input_data.resize((1, input_data.size))
            inputs.append(input_data[0])
    # 整个神经网络学习结果由两个（可修改成其他深度）权重矩阵（主要）构成
    input_layer_col=input_data.size
    inner_layer_row=input_layer_col
    inner_layer_col=1000
    output_layer_row=inner_layer_col
    output_layer_col=1
    w1=2*np.random.random((inner_layer_row, inner_layer_col)) - 1
    w2=2*np.random.random((output_layer_row, output_layer_col)) - 1

    x=np.array(inputs)
    y=np.array(out_puts)
    y.resize((y.size,1))


    # BP实现
    for j in range(2000):
        #正常计算网络各层各节点的值
        #正常计算网络各层各节点的值
        l1=logistic(np.dot(x, w1))
        l2=logistic(np.dot(l1, w2))

        #从后向前计算每层误差以及高确信误差
        #前层的误差由后层的高确信误差、该层与后层的权重网络决定
        l2_error=y-l2

        #以下是输出误差提示
        if(j%50)==0:
            print("Error"+str(np.mean(np.abs(l2_error))))

        if (np.mean(np.abs(l2_error)) < 0.0001):
            break

        l2_delta=l2_error*logistic(l2, True)

        l1_error=l2_delta.dot(w2.T)

        l1_delta=l1_error*logistic(l1, True)

        w2 = w2 + l1.T.dot(l2_delta) / 100
        w1 = w1 + x.T.dot(l1_delta) / 100

    print(j)
    print("ERR:\n",l2_error)
    # 保存训练好的网络
    saveWeights(w1, "w1")
    saveWeights(w2, "w2")


def predictMatchIndex():
    w1 = readWeights("w1.txt")
    w2 = readWeights("w2.txt")
    # 加载标准输入数据，并进行矩阵拼接，将两张图片拼接在一起作为一个输入
    input_data_list, _ = readStandardData("std_ikx")
    input_data_list_2, _ = readStandardData("sb")

    np.random.seed(1)

    same_predict = 0
    not_same_predict = 0
    for i in range(40):

        # 拼接 竖向
        # 不同类分类情况
        j = np.random.randint(0, len(input_data_list))
        k = np.random.randint(0, len(input_data_list))
        input_data = np.vstack((input_data_list[j],input_data_list_2[k]))
        # cv2.imshow("PIC", input_data)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        input_data.resize((1,input_data.size))

        l1 = logistic_function(np.dot(input_data, w1))
        l2 = logistic_function(np.dot(l1, w2))

        # 输出0是预测同类
        # 输出1是预测不相同
        if l2 < 0.6:
            same_predict = same_predict+1
        else:
            not_same_predict = not_same_predict+1

    # **********************************
    # 输出出现全激活或全不激活情况！！！！！！
    # ！！！！！！！！！！！！！！！！！！！！
    # **********************************
    print("input", input_data)
    print("pre_l1", (np.dot(input_data, w1)))
    print("l1", l1)
    print("l2", l2)
    print("Right Rate: ", str(not_same_predict/40))

def saveWeights(weights_vars, file_nmae):
    writefile = open((file_nmae + ".txt"), "w")
    for i in weights_vars:
        for j in i:
            writefile.write(str(j)+" ")
        writefile.write("\n")
    writefile.close()


#非线性映射函数，到0~1的范围；及其导数，当deriv为True时
def nonlin(x, derive=False):
    y=np.tanh(x)
    if derive:
        return (1-y**2)
    return y

def logistic(x, derive=False):
    y = 1/(1 + np.exp(-x))
    if derive:
        return y * (1 - y)
    else:
        return y

def logistic_function(x, derive=False):
    y = .5 * (1 + np.tanh(.5 * x))
    if derive:
        return y * (1 - y)
    else:
        return y

def addLayer(layers, current_layer):
    weights = np.array[layers[current_layer-1],layers[current_layer]]
    bias = np.array[current_layer]


if __name__ == "__main__":
    # trainNeuralNetwork()
    predictMatchIndex()