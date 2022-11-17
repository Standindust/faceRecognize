import numpy as np
import argparse
import cv2 as cv

import pyttsx3
import os
import urllib
from PIL import Image, ImageDraw, ImageFont

import webcam
engine = pyttsx3.init()
#加载训练数据集文件
recogizer=cv.face.LBPHFaceRecognizer_create()
#读取训练好的系统文件

#存储人脸库中人员的名字
roomnumbers=[]
#对应的标签
idn = []

alien= {}


types=[]
inWidth = 300
inHeight = 300
confThreshold = 0.5
count = 0

# prototxt = 'face_detector/deploy.prototxt'  # 调用.caffemodel时的测试网络文件
# caffemodel = 'face_detector/res10_300x300_ssd_iter_140000.caffemodel'  # 包含实际图层权重的.caffemodel文件
net = cv.dnn.readNetFromTensorflow("face_detector/opencv_face_detector_uint8.pb","face_detector/opencv_face_detector.pbtxt")

def face_detection(frame):
    # net = cv.dnn.readNetFromCaffe(prototxt, caffemodel)


    # cap = cv.VideoCapture("E:/视频库/srcImage/OneStopMoveEnter1cor.avi")
    # rame = cv.imread("face01.jpg")
    cols = frame.shape[1]
    rows = frame.shape[0]

    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (104.0, 177.0, 123.0), False, False))
    detections = net.forward()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confThreshold:

            xLeftBottom = int(detections[0, 0, i, 3] * cols)
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop = int(detections[0, 0, i, 5] * cols)
            yRightTop = int(detections[0, 0, i, 6] * rows)
            if xLeftBottom<0 :xLeftBottom=1
            if xRightTop<0 :xRightTop=1
            if yLeftBottom<0:xLeftBottom=1
            if yRightTop<0 :yRightTop=1
            

            if xRightTop-xLeftBottom >300 and yRightTop-yLeftBottom>300:
                cv.putText(frame, "stay back", (0, 200), cv.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 8)
                cv.imshow("detections", frame)
                break
            cv.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop), color=(0, 255, 0), thickness=2)
            global confidence1
            global ids
            try:
                ids, confidence1 = recogizer.predict(gray[yLeftBottom:yRightTop, xLeftBottom:xRightTop])
            except:
                cv.putText(frame, "Keep center", (0, 200), cv.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 8)
                cv.imshow("detections", frame)
                print(str(xLeftBottom)+"\n"+str(xRightTop)+"\n"+str(yLeftBottom)+"\n"+str(yRightTop))

            if confidence1 > 45:
                frame = cv2AddChineseText(frame, "外来人员", (xLeftBottom, yLeftBottom - 20), (0, 255, 0), 25)

            # putText(img, 'unkonw', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
            else:
                title = '房间号:'
                if roomnumbers[idn.index(alien.get(ids))] == 1:
                    title = '客房部:'
                if roomnumbers[idn.index(alien.get(ids))] == 2:
                    title = '人力资源部:'
                if roomnumbers[idn.index(alien.get(ids))] == 3:
                    title = '迎宾部:'
                if roomnumbers[idn.index(alien.get(ids))] == 4:
                    title = '保洁部:'

                frame = cv2AddChineseText(frame, title + str(roomnumbers[idn.index(alien.get(ids))]) +
                                          "\nID:" + alien.get(ids) +
                                          "\nType:" + types[idn.index(alien.get(ids))] + "\nids:" + str(confidence1),

                                          (xLeftBottom, yLeftBottom - 80), (0, 255, 0), 25)


        cv.imshow("detections", frame)











def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB
                                           ))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)
def name():
    path = 'D:\\1pyidentity\\imgdata\\allface'
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    for imagePath in imagePaths:
       id = str(os.path.split(imagePath)[1].split('.',3)[1])
       id1 = int(id[0]+id[5]+id[6]+id[9]+id[11]+id[13]+id[15])

       if alien.get(id) ==None:
           alien[id1] = id


       roomnumber = int(os.path.split(imagePath)[1].split('.',3)[0])
       type = str(os.path.split(imagePath)[1].split('.')[3])
       roomnumbers.append(roomnumber)
       idn.append(id)
       types.append(type)
cap = cv.VideoCapture(0)
name()
num=0
recogizer.read('D:\\1pyidentity\\trainer\\allface\\trainer.yml')


while True:
        if num == 1000:
            recogizer.read('D:\\1pyidentity\\trainer\\allface\\trainer.yml')
            print("LOAD")
            num = 0
        ret, frame = cap.read()
        face_detection(frame)
        if ord(' ') == cv.waitKey(10):
            break
        num = num + 1

cv.destroyAllWindows()
cap.release()




