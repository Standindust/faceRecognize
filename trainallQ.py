import os
import sys

import cv2 as cv
from PIL import Image
import numpy as np

# prototxt = 'face_detector/deploy.prototxt'  # 调用.caffemodel时的测试网络文件
# caffemodel = 'face_detector/res10_300x300_ssd_iter_140000.caffemodel'  # 包含实际图层权重的.caffemodel文件
inWidth = 300
inHeight = 300
confThreshold = 0.9
net = cv.dnn.readNetFromTensorflow("face_detector/opencv_face_detector_uint8.pb","face_detector/opencv_face_detector.pbtxt")

def getImageAndlabels(path):
    # 人脸数据数据
    facesSamples = []
    # 人标签
    ids = []
    #房间号标签
    rooms = []
    #类型标签
    types = []
    # 读取所有的照片的名称（os.listdir读取根目录下文件的名称返回一个列表，os.path.join将根目录和文件名称组合形成完整的文件路径）
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # 调用人脸分类器（注意自己文件保存的路径，英文名）
    # face_detect = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    # 循环读取照片人脸数据
    # net = cv.dnn.readNetFromCaffe(prototxt, caffemodel)
    # net = cv.dnn.readNetFromTensorflow("face_detector/opencv_face_detector_uint8.pb","face_detector/opencv_face_detector.pbtxt")

    count=1
    for imagePath in imagePaths:
        # 用灰度的方式打开照片
        # PIL_img = Image.open(imagePath).convert('L')
        # # 将照片转换为计算机能识别的数组OpenCV（BGR--0-255）
        # img_numpy = np.array(PIL_img, 'uint8')
        frame = cv.imread(imagePath)
        # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #
        # faces = face_detect.detectMultiScale(img_numpy)

        id = str(os.path.split(imagePath)[1].split('.')[1])
        id = int(id[0] + id[5] + id[6] + id[9] + id[11] + id[13] + id[15])
        # cols = frame.shape[1]
        # rows = frame.shape[0]
        # net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth,inHeight), (104.0, 177.0, 123.0), False, False))
        # detections = net.forward()
        #
        # # perf_stats = net.getPerfProfile()
        # # print('Inference time:  %.2f ms' % (perf_stats[0] / cv.getTickFrequency() * 1000))
        #
        # for i in range(detections.shape[2]):
        #     confidence = detections[0, 0, i, 2]
        #     if confidence > confThreshold:
        #         print(confidence)
        #
        #         xLeftBottom = int(detections[0, 0, i, 3] * cols)
        #         yLeftBottom = int(detections[0, 0, i, 4] * rows)
        #         xRightTop = int(detections[0, 0, i, 5] * cols)
        #         yRightTop = int(detections[0, 0, i, 6] * rows)
        #         ids.append(id)
        #         gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #         facesSamples.append(gray[yLeftBottom:yRightTop,xLeftBottom:xRightTop])
        #
        #         count=count+1
        #         cv.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop), color=(0, 255, 0), thickness=2)
        #         cv.imshow("detections2", gray[yLeftBottom:yRightTop, xLeftBottom:xRightTop])
        #         cv.imshow("detections", frame)
        #
        #
        #         if cv.waitKey(0) != -1:
        #             break
        #         print(imagePath)



        ids.append(id)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        facesSamples.append(gray)

        count = count + 1
        print(imagePath)
        # cv.imshow("detections", gray)
        # if cv.waitKey(0) != -1:
        #         break
    return facesSamples, ids









if __name__ == '__main__':
    # 人脸图片存放的文件夹

    path = 'D:\\1pyidentity\\imgdata\\allface'
    faces, ids = getImageAndlabels(path)
    # 调用LBPH算法对人脸数据进行处理
    recognizer = cv.face.LBPHFaceRecognizer_create()
    # 训练数据
    recognizer.train(faces, np.array(ids))

    # 将训练的系统保存在特定文件夹
    path = r'D:\\1pyidentity\\trainer\\allface'

    if (os.path.exists(path) == 0):
        os.mkdir(r'D:\\1pyidentity\\trainer\\allface')

    recognizer.write('D:\\1pyidentity\\trainer\\allface\\trainer.yml')
    print("finish");