import argparse
import os

import random
import shutil
import sys

import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
import cv2 as cv
import numpy as np
import align_my


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
        # 用灰度的方式打开照片

        img = cv.imread(imagePath)
        id = str(os.path.split(imagePath)[1].split('.')[1])
        id = int(id[0] + id[5] + id[6] + id[9] + id[11] + id[13] + id[15])
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = def_equalizehist(gray)
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
        print("检测面部图片：" + imagePath + "\t训练中")

    return facesSamples, ids

def salt_and_pepper_noise(img, percentage):
    rows, cols = img.shape
    num = int(percentage * rows * cols)
    for i in range(num):
        x = random.randint(0, rows - 1)
        y = random.randint(0, cols - 1)
        if random.randint(0, 1) == 0:
            img[x, y] = 0  # 黑色噪声
        else:
            img[x, y] = 255  # 白色噪声
    return img

    # 添加高斯噪声

#
# @numba.vectorize(['int64(int64, int64,int64,int64)'], target='cuda')


def def_equalizehist(img, L=256):


    h, w = img.shape
    # 计算图像的直方图，即存在的每个灰度值的像素点数量
    hist = cv.calcHist([img], [0], None, [256], [0, 255])
    # 计算灰度值的像素点的概率，除以所有像素点个数，即归一化
    hist[0:255] = hist[0:255] / (h * w)
    # 设置Si
    sum_hist = np.zeros(hist.shape)
    # 开始计算Si的一部分值，注意i每增大，Si都是对前i个灰度值的分布概率进行累加
    for i in range(256):
        sum_hist[i] = sum(hist[0:i + 1])
    equal_hist = np.zeros(sum_hist.shape)
    # Si再乘上灰度级，再四舍五入
    for i in range(256):
        equal_hist[i] = int(((L - 1) - 0) * sum_hist[i] + 0.5)
    equal_img = img.copy()
    # 新图片的创建
    for i in range(h):
        for j in range(w):
            equal_img[i, j] = equal_hist[img[i, j]]

    equal_hist = cv.calcHist([equal_img], [0], None, [256], [0, 256])
    equal_hist[0:255] = equal_hist[0:255] / (h * w)
    # cv.imshow("inverse", equal_img)
    # 显示最初的直方图
    # plt.figure("原始图像直方图")
    plt.plot(hist, color='b')
    # plt.show()
    # plt.figure("直方均衡化后图像直方图")
    plt.plot(equal_hist, color='r')
    # plt.show()
    # cv.waitKey()
    # return equal_hist
    return equal_img


def gaussian_noise(img, mu, sigma, k):
    rows, cols = img.shape
    # def goj(j,cols):
    #     for j in range(cols):
    #         # 生成高斯分布的随机数，与原始数据相加后要取整
    #         value = int(img[i, j] + k * random.gauss(mu=mu, sigma=sigma))
    #         # 限定数据值的上下边界
    #         value = np.clip(a_max=255, a_min=0, a=value)
    #         img[i, j] = value
    for i in range(rows):
        # thread1 = threading.Thread(name='t1', target=goj, args=(0,int(cols*(1/5))))
        # thread2 = threading.Thread(name='t2', target=goj, args=(int(cols*(1/5))+1, int(cols*(2/5))))
        # thread3 = threading.Thread(name='t2', target=goj, args=(int(cols * (2 / 5) )+ 1, int(cols * (3 / 5))))
        # thread4 = threading.Thread(name='t2', target=goj, args=(int(cols * (3 / 5) )+ 1, int(cols * (4 / 5))))
        # thread5 = threading.Thread(name='t2', target=goj, args=(int(cols * (4 / 5) )+ 1, int(cols * (5 / 5))))
        #
        # thread1.start()  # 启动线程1
        # thread2.start()  # 启动线程2
        # thread3.start()  # 启动线程2
        # thread4.start()  # 启动线程2
        # thread5.start()  # 启动线程2
        for j in range(cols):
            # 生成高斯分布的随机数，与原始数据相加后要取整
            value = int(img[i, j] + k * random.gauss(mu=mu, sigma=sigma))
            # 限定数据值的上下边界
            value = np.clip(a_max=255, a_min=0, a=value)
            img[i, j] = value
    return img

# @vectorize(['int64(int64, int64,int64)'], target='cuda')

def yuan_LBP(img, r=3, p=8):
    h, w = img.shape
    dst = np.zeros((h, w), dtype=img.dtype)
    for i in range(r, h - r):
        for j in range(r, w - r):
            LBP_str = []
            for k in range(p):
                rx = i + r * np.cos(2 * np.pi * k / p)
                ry = j - r * np.sin(2 * np.pi * k / p)
                # print(rx, ry)
                x0 = int(np.floor(rx))
                x1 = int(np.ceil(rx))
                y0 = int(np.floor(ry))
                y1 = int(np.ceil(ry))

                f00 = img[x0, y0]
                f01 = img[x0, y1]
                f10 = img[x1, y0]
                f11 = img[x1, y1]
                w1 = x1 - rx
                w2 = rx - x0
                w3 = y1 - ry
                w4 = ry - y0
                fxy = w3 * (w1 * f00 + w2 * f10) + w4 * (w1 * f01 + w2 * f11)
                if fxy >= img[i, j]:
                    LBP_str.append(1)
                else:
                    LBP_str.append(0)
            # temp=""
            # for i in range(len(LBP_str)):
            #     temp+=str(LBP_str[i])
            # LBP_str = temp
            LBP_str = ''.join('%s' % id for id in LBP_str)
            dst[i, j] = int(LBP_str, 2)
    return dst
def parse_arguments(argv,indir,outdir):
    parser = argparse.ArgumentParser()

    parser.add_argument('--indir',
                        default=indir,
                        type=str, help='Directory with unaligned images.')
    parser.add_argument('--outdir',
                        default=outdir,
                        type=str, help='Directory with aligned face thumbnails.')

    parser.add_argument('--image-size', type=str, help='Image size (height, width) in pixels.', default='448,448')
    # parser.add_argument('--margin', type=int,
    #    help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)

    parser.add_argument('--exts', nargs='+', default=['.jpeg', '.jpg', '.png','.bmp'],
                        help='list of acceptable image extensions.')

    parser.add_argument('--recursive', default=True,
                        help='If true recursively walk through subdirs and assign an unique label\
        to images in each folder. Otherwise only include images in the root folder\
        and give them label 0.')
    return parser.parse_args(argv)

def copy(path, path1):
    print("copy")
    if (os.path.exists(path1) == 0):
        os.mkdir(path1)
    for file in os.listdir(path):
            # 遍历原文件夹中的文件
        full_file_name = os.path.join(path, file)  # 把文件的完整路径得到
            # print("要被复制的全文件路径全名:", full_file_name)
        if os.path.isfile(full_file_name):  # 用于判断某一对象(需提供绝对路径)是否为文件
            shutil.copy(full_file_name, path1)  # shutil.copy函数放入原文件的路径文件全名  然后放入目标文件夹
    print("copy finish")



if __name__ == '__main__':


    # 人脸图片存放的文件夹
    # temp=sys.argv[1:]
    # type = temp.pop()
    # file =temp.pop()

    argv=[]
    file ='1514'
    type ='guest'
    path = "D:\\1pyidentity/imgdata/"+type+'/'+file+".redoimg"
    path1="D:\\1pyidentity/imgdata/"+type+'/'+file
    des=r"D:\1pyidentity\imgdata\allface\\"+file
    # path ="./redoimg"
    parser=parse_arguments(argv, path1, path)
    align_my.main(parser)
    copy(path,des)
    faces, ids = getImageAndlabels(path)
    # 调用LBPH算法对人脸数据进行处理
    recognizer = cv.face.LBPHFaceRecognizer_create()
    # 训练数据
    recognizer.train(faces, np.array(ids))
    # 将训练的系统保存在特定文件夹
    path = r'D:\\1pyidentity\\trainer\\' + file

    if (os.path.exists(path) == 0):
        os.mkdir(r'D:\\1pyidentity\\trainer\\'+file)

    recognizer.write('D:\\1pyidentity\\trainer\\'+file+'\\trainer.yml')
    # recognizer.write('./trainer/trainer.yml')

    print("train finish")
