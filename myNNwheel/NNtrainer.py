#-*- coding: UTF-8 -*-
import numpy as np

def trainNeuralNetwork(PathList=["data_1.txt","data_2.txt","data_3.txt","data_4.txt"]):
    #_________________数据与代码同一目录时用下列代码_________________________

    # 加载标准输入数据


    # 快速生成标准输出矩阵（即分类标记）


    # 整个学习网络由两个（可修改成其他深度）权重矩阵（主要）构成，

    # BP实现
    for j in range(60000):
        #正常计算网络各层各节点的值
        pass


    # 保存训练好的网络



def saveWeights(WeightsVars,FileNmae):
    writefile = open((FileNmae+".txt"), "w")
    for i in WeightsVars:
        for j in i:
            writefile.write(str(j)+" ")
        writefile.write("\n")
    writefile.close()


#非线性映射函数，到0~1的范围；及其导数，当deriv为True时
def nonlin(x,deriv=False):
    y=np.tanh(x)
    if deriv:
        return (1-y**2)
    return y

def add_layer():
    pass
