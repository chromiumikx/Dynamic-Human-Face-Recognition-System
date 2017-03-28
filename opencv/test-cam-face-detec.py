import cv2
import numpy as np
from PIL import Image,ImageDraw

def detectFaces(image_name):
    img = cv2.imread(image_name)
    face_cascade = cv2.CascadeClassifier("F:\\Cache\\GitHub\\Dynamic-Human-Face-Recognition-System\\opencv\\haarcascade_frontalface_alt.xml")
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img #if语句：如果img维度为3，说明不是灰度图，先转化为灰度图gray，如果不为3，也就是2，原图就是灰度图

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)#1.3和5是特征的最小、最大检测窗口，它改变检测结果也会改变
    result = []
    for (x,y,width,height) in faces:
        result.append((x,y,x+width,y+height))
    return result


def drawFaces(image_name):
    faces = detectFaces(image_name)
    if faces:
        img = Image.open(image_name)
        draw_instance = ImageDraw.Draw(img)
        for (x1,y1,x2,y2) in faces:
            draw_instance.rectangle((x1,y1,x2,y2), outline=(255, 0,0))
        img.show()
        img.save('drawfaces_'+image_name)


# drawFaces('abc.jpg')

cap = cv2.VideoCapture(0)
 
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
 
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 载入分类器
    face_cascade = cv2.CascadeClassifier("F:\\Cache\\GitHub\\Dynamic-Human-Face-Recognition-System\\opencv\\haarcascade_frontalface_alt.xml")
 
    # Display the resulting frame
    # cv2.imshow('frame',gray)

    # 识别人脸输出坐标
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)#1.3和5是特征的最小、最大检测窗口，它改变检测结果也会改变
    result = []
    for (x,y,width,height) in faces:
        result.append((x,y,x+width,y+height))

    if result:
        for (x1,y1,x2,y2) in result:
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
    else:
        pass
    cv2.imshow('frame',frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()    
