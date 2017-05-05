import cv2
import matplotlib.pyplot as plt


def now_state(image_size, learning_rate, conv_layers, fc_layers, dropout_keep_prob):
    return "imsize"+str(image_size)+"lrate"+str(learning_rate)+"convs"+str(conv_layers)+"fcs"+str(fc_layers)+"kpb"+str(dropout_keep_prob)


def save_result(result_var, result_name, end_add):
    f = open(result_name+end_add, "r")

