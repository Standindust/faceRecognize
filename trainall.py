import os
import sys

import cv2 as cv
from PIL import Image
import numpy as np
from skimage.feature import local_binary_pattern

import train

def getImageAndlabels(path):
    # 人脸数据数据
    facesSamples = []
    # 人标签
    ids = []
    # 读取所有的照片的名称（os.listdir读取根目录下文件的名称返回一个列表，os.path.join将根目录和文件名称组合形成完整的文件路径）
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # 调用人脸分类器（注意自己文件保存的路径，英文名）

    print("getImageAndlabels\n")
    # 循环读取照片人脸数据
    count = 1
    for imagePath in imagePaths:
        imagePath1 = [os.path.join(imagePath, f) for f in os.listdir(imagePath)]

        for imagePath2 in imagePath1:

            img = cv.imread(imagePath2)
            id = str(os.path.split(imagePath2)[1].split('.')[1])
            id = int(id[0] + id[5] + id[6] + id[9] + id[11] + id[13] + id[15])
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            gray = train.def_equalizehist(gray)
            gray1 = gray.copy()
            # 直接去椒盐
            r = 2  # 邻域半径
            p = 8 * r  # 邻域采样点数量
            # gray1 = cv.medianBlur(gray1, 5)
            # 直接去椒盐高斯

            gray1 = cv.GaussianBlur(gray1, (5, 5), 0)
            uniformLBP = local_binary_pattern(gray1, p, r, method="uniform")
            ids.append(id)
            facesSamples.append(uniformLBP)
            count = count + 1
            print("检测面部图片：" + imagePath2 + "\t训练中")

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
def run():
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
