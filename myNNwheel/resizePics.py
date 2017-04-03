# 复制到图片所在的文件夹中运行
#
import os
import cv2

if __name__ == "__main__":
    pic_names = os.listdir()
    pic_names.pop()
    print(pic_names)

    for ipic in pic_names:
        img = cv2.imread(ipic)# 以灰度模式读取标准数据
        std_iface = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
        if std_iface.ndim == 3:
            gray = cv2.cvtColor(std_iface, cv2.COLOR_BGR2GRAY)
        else:
            gray = std_iface
        cv2.imwrite(ipic, gray)