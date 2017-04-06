#-*- coding: UTF-8 -*-
import cv2
import numpy as np
from readDataset import readStandardData
from readDataset import readWeights

def trainNeuralNetwork():
    #_________________数据与代码同一目录时用下列代码_________________________

    np.random.seed(1)

    '''
    标准数据拼接、加载
    '''
    # 加载标准输入数据，并进行矩阵拼接，将两张图片拼接在一起作为一个输入
    train_pics, train_user_id = readStandardData(["ikx", "qin"])

    train_samples = []
    train_labels = []
    two_pics = []
    for i in range(len(train_pics)):
        for j in range(len(train_pics)):
            # 输出 [0,1] 表示是相同
            if train_user_id[i] == train_user_id[j]:
                # 两张图片位置再对调
                train_labels.append([0, 1.])
                train_labels.append([0, 1.])
            else:
            # 输出 [1,0] 表示相异
                train_labels.append([1., 0])
                train_labels.append([1., 0])
            # 拼接 竖向
            two_pics = np.vstack((train_pics[i], train_pics[j]))
            # .resize()是一个操作型函数
            two_pics.resize((1, two_pics.size))
            train_samples.append(two_pics[0])
            two_pics = np.vstack((train_pics[j], train_pics[i]))
            # .resize()是一个操作型函数
            two_pics.resize((1, two_pics.size))
            train_samples.append(two_pics[0])


    '''
    层数、神经元数目
    '''
    # 每层的神经元数目
    # 第零层 输入的长度
    # 第i层大小
    layers = [two_pics.size, 1000, 2]


    '''
    权重
    '''
    Weights = []
    bias = []
    for i in range(len(layers)-1):
        Weights.append(2 * np.random.random((layers[i], layers[i + 1])) - 1)

    Weights = np.array(Weights)


    '''
    模型：整个运算过程
    '''
    patch_size = 400
    learn_rate = 1/100.
    x=np.array(train_samples)
    y=np.array(train_labels)
    print(x.shape)
    print(y.shape)
    for j in range(100):
        for x_, y_ in getPatch(x, y, patch_size):
            #正常计算网络各层各节点的值
            #正常计算网络各层各节点的值
            l1=logistic(np.dot(x_, Weights[0]))
            l2=logistic(np.dot(l1, Weights[1]))

            #从后向前计算每层误差以及高确信误差
            #前层的误差由后层的高确信误差、该层与后层的权重网络决定
            l2_error=y_-l2

            if (np.mean(np.abs(l2_error)) < 0.000001):
                break

            l2_delta=l2_error*logistic(l2, True)

            l1_delta=(l2_delta.dot(Weights[1].T))*logistic(l1, True)

            '''
            梯度下降算法更新权重
            '''
            Weights[1] = Weights[1] + (l1.T.dot(l2_delta)) * learn_rate
            Weights[0] = Weights[0] + (x_.T.dot(l1_delta)) * learn_rate

        #以下是输出误差提示
        if(j%10)==0:
            print("Error"+str(np.mean(np.abs(l2_error))))
            print("准确：", np.sum(np.argmax(y_, 1) == np.argmax(l2, 1)) / patch_size)

    l1 = logistic(np.dot(x, Weights[0]))
    l2 = logistic(np.dot(l1, Weights[1]))
    print(np.sum(np.argmax(y, 1) == np.argmax(l2, 1))/y.shape[0])


    '''
    测试、数据拼接加载、运算
    '''
    # 加载标准输入数据，并进行矩阵拼接，将两张图片拼接在一起作为一个输入
    test_pics, test_user_id = readStandardData(["sb"])

    test_samples = []
    test_labels = []
    test_two_pics = []
    for i in range(len(test_pics)):
        for j in range(len(test_pics)):
            # 输出 [0,1] 表示是相同
            if test_user_id[i] == test_user_id[j]:
                test_labels.append([0, 1.])
            else:
            # 输出 [1,0] 表示相异
                test_labels.append([1., 0])
            # 拼接 竖向
            test_two_pics = np.vstack((test_pics[i], test_pics[j]))
            # .resize()是一个操作型函数
            test_two_pics.resize((1, test_two_pics.size))
            test_samples.append(test_two_pics[0])

    x=np.array(test_samples)
    y=np.array(test_labels)
    l1 = logistic(np.dot(x, Weights[0]))
    l2 = logistic(np.dot(l1, Weights[1]))
    print("Use sb:", np.sum(np.argmax(y, 1) == np.argmax(l2, 1)) / y.shape[0])


    # 保存训练好的网络
    # saveWeights(Weights[0], "w1")
    # saveWeights(Weights[1], "w2")


def getPatch(x, y, patch_size):
    step_start = 0
    while step_start < len(x):
        step_end = step_start + patch_size
        if step_end < len(x):
            yield x[step_start:step_end], y[step_start:step_end]
        step_start = step_end


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
    pass


if __name__ == "__main__":
    trainNeuralNetwork()
