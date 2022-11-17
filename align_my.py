from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
# import facenet
import detect_face
import random
from time import sleep

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
import face_image
import face_preprocess as face_preprocess
from skimage import transform as trans
import cv2

import os

import random

import matplotlib.pyplot as plt


import cv2 as cv
from PIL import Image
import numpy as np
import threading

from numba import cuda, numba

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def IOU(Reframe, GTframe):
    x1 = Reframe[0];
    y1 = Reframe[1];
    width1 = Reframe[2] - Reframe[0];
    height1 = Reframe[3] - Reframe[1];

    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2] - GTframe[0]
    height2 = GTframe[3] - GTframe[1]

    endx = max(x1 + width1, x2 + width2)
    startx = min(x1, x2)
    width = width1 + width2 - (endx - startx)

    endy = max(y1 + height1, y2 + height2)
    starty = min(y1, y2)
    height = height1 + height2 - (endy - starty)

    if width <= 0 or height <= 0:
        ratio = 0
    else:
        Area = width * height
        Area1 = width1 * height1
        Area2 = width2 * height2
        ratio = Area * 1. / (Area1 + Area2 - Area)
    return ratio
def list_image(root, recursive, exts):
    """Traverses the root of directory that contains images and
    generates image list iterator.
    Parameters
    ----------
    root: string
    recursive: bool
    exts: string
    Returns
    -------
    image iterator that contains all the image under the specified path
    """

    i = 0
    if recursive:
        cat = {}
        for path, dirs, files in os.walk(root, followlinks=True):
            dirs.sort()
            files.sort()
            for fname in files:
                fpath = os.path.join(path, fname)
                suffix = os.path.splitext(fname)[1].lower()
                if os.path.isfile(fpath) and (suffix in exts):
                    if path not in cat:
                        cat[path] = len(cat)
                    yield (i, os.path.relpath(fpath, root), cat[path])
                    i += 1
        for k, v in sorted(cat.items(), key=lambda x: x[1]):
            print(os.path.relpath(k, root), v)
    else:
        for fname in sorted(os.listdir(root)):
            fpath = os.path.join(root, fname)
            suffix = os.path.splitext(fname)[1].lower()
            if os.path.isfile(fpath) and (suffix in exts):
                yield (i, os.path.relpath(fpath, root), 0)
                i += 1

def main(args):
    imgs_info = list_image(args.indir, args.recursive, args.exts)
    #print(imgs_info)

    print('Creating networks and loading parameters')

    with tf.Graph().as_default():
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess = tf.compat.v1.Session()
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    minsize = 60
    threshold = [0.6, 0.85, 0.8]
    factor = 0.85

    # Add a random key to the filename to allow alignment using multiple processes
    # random_key = np.random.randint(0, high=99999)
    # bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)
    # output_filename = os.path.join(output_dir, 'faceinsight_align_%s.lst' % args.name)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    output_filename = os.path.join(args.outdir, 'lst')

    nrof_images_total = 0
    nrof = np.zeros((5,), dtype=np.int32)
    face_count = 0
    for src_img_info in imgs_info:
        src_img_path = os.path.join(args.indir,src_img_info[1])
        if nrof_images_total % 100 == 0:
            print("Processing %d, (%s)" % (nrof_images_total, nrof))
        nrof_images_total += 1
        if not os.path.exists(src_img_path):
            print('image not found (%s)' % src_img_path)
            continue
        # print(image_path)
        try:
            img = misc.imread(src_img_path)
        except (IOError, ValueError, IndexError) as e:
            errorMessage = '{}: {}'.format(src_img_path, e)
            print(errorMessage)
        else:
            if img.ndim < 2:
                print('Unable to align "%s", img dim error' % src_img_path)
                # text_file.write('%s\n' % (output_filename))
                continue
            if img.ndim == 2:
                img = to_rgb(img)
            img = img[:, :, 0:3]

            target_dir = ''
            src_img_path_list = src_img_info[1].replace('\\','/').split('/')
            #print(src_img_path_list)
            if len(src_img_path_list) <= 1:
                target_dir = args.outdir
            elif len(src_img_path_list) ==2:
                src_img_path_prefix = src_img_path_list[:-1]
                target_dir = os.path.join(args.outdir, str(src_img_path_prefix[0]))
            else:
                src_img_path_prefix = ['/'.join(bar) for bar in src_img_path_list[:-1]]

                #print('src_img_path_prefix: ',src_img_path_prefix)
                target_dir = os.path.join(args.outdir, str(src_img_path_prefix[0]))
            #print(target_dir)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            _minsize = minsize
            _bbox = None
            _landmark = None

            bounding_boxes, points = detect_face.detect_face(img, _minsize, pnet, rnet, onet, threshold, factor)
            print("done:"+src_img_path)

            #print(points)
            if points == []:
                print(src_img_path)
            else:
                _landmark = points.T

            #print(_landmark.shape)
            faces_sumnum = bounding_boxes.shape[0]
            for  num  in  range(faces_sumnum):
                # cv2.imwrite(target_dir+r"/"+src_img_path_list[0],img)
                warped = face_preprocess.preprocess(img, bbox=bounding_boxes[num], landmark=_landmark[num].reshape([2,5]).T, image_size=args.image_size)
                # cv2.imwrite(target_dir + r"/" + src_img_path_list[0], warped)
                bgr = warped[..., ::-1]


                # target_file = os.path.join(target_dir, '%04d.jpg' % face_count)
                # cv2.imwrite(target_file, bgr)
                cv2.imwrite(target_dir+r"/"+src_img_path_list[0],bgr)
                face_count += 1
#
# # 添加椒盐噪声
# def salt_and_pepper_noise(img, percentage):
#     rows, cols = img.shape
#     num = int(percentage * rows * cols)
#     for i in range(num):
#         x = random.randint(0, rows - 1)
#         y = random.randint(0, cols - 1)
#         if random.randint(0, 1) == 0:
#             img[x, y] = 0  # 黑色噪声
#         else:
#             img[x, y] = 255  # 白色噪声
#     return img
#
#     # 添加高斯噪声
#
# #
# # @numba.vectorize(['int64(int64, int64,int64,int64)'], target='cuda')
#
#
# def def_equalizehist(img, L=256):
#
#
#     h, w = img.shape
#     # 计算图像的直方图，即存在的每个灰度值的像素点数量
#     hist = cv2.calcHist([img], [0], None, [256], [0, 255])
#     # 计算灰度值的像素点的概率，除以所有像素点个数，即归一化
#     hist[0:255] = hist[0:255] / (h * w)
#     # 设置Si
#     sum_hist = np.zeros(hist.shape)
#     # 开始计算Si的一部分值，注意i每增大，Si都是对前i个灰度值的分布概率进行累加
#     for i in range(256):
#         sum_hist[i] = sum(hist[0:i + 1])
#     equal_hist = np.zeros(sum_hist.shape)
#     # Si再乘上灰度级，再四舍五入
#     for i in range(256):
#         equal_hist[i] = int(((L - 1) - 0) * sum_hist[i] + 0.5)
#     equal_img = img.copy()
#     # 新图片的创建
#     for i in range(h):
#         for j in range(w):
#             equal_img[i, j] = equal_hist[img[i, j]]
#
#     equal_hist = cv2.calcHist([equal_img], [0], None, [256], [0, 256])
#     equal_hist[0:255] = equal_hist[0:255] / (h * w)
#     # cv.imshow("inverse", equal_img)
#     # 显示最初的直方图
#     # plt.figure("原始图像直方图")
#     plt.plot(hist, color='b')
#     # plt.show()
#     # plt.figure("直方均衡化后图像直方图")
#     plt.plot(equal_hist, color='r')
#     # plt.show()
#     # cv.waitKey()
#     # return equal_hist
#     return equal_img
#
#
# def gaussian_noise(img, mu, sigma, k):
#     rows, cols = img.shape
#     # def goj(j,cols):
#     #     for j in range(cols):
#     #         # 生成高斯分布的随机数，与原始数据相加后要取整
#     #         value = int(img[i, j] + k * random.gauss(mu=mu, sigma=sigma))
#     #         # 限定数据值的上下边界
#     #         value = np.clip(a_max=255, a_min=0, a=value)
#     #         img[i, j] = value
#     for i in range(rows):
#         # thread1 = threading.Thread(name='t1', target=goj, args=(0,int(cols*(1/5))))
#         # thread2 = threading.Thread(name='t2', target=goj, args=(int(cols*(1/5))+1, int(cols*(2/5))))
#         # thread3 = threading.Thread(name='t2', target=goj, args=(int(cols * (2 / 5) )+ 1, int(cols * (3 / 5))))
#         # thread4 = threading.Thread(name='t2', target=goj, args=(int(cols * (3 / 5) )+ 1, int(cols * (4 / 5))))
#         # thread5 = threading.Thread(name='t2', target=goj, args=(int(cols * (4 / 5) )+ 1, int(cols * (5 / 5))))
#         #
#         # thread1.start()  # 启动线程1
#         # thread2.start()  # 启动线程2
#         # thread3.start()  # 启动线程2
#         # thread4.start()  # 启动线程2
#         # thread5.start()  # 启动线程2
#         for j in range(cols):
#             # 生成高斯分布的随机数，与原始数据相加后要取整
#             value = int(img[i, j] + k * random.gauss(mu=mu, sigma=sigma))
#             # 限定数据值的上下边界
#             value = np.clip(a_max=255, a_min=0, a=value)
#             img[i, j] = value
#     return img

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

#
if __name__ == '__main__':
    argv=[]
    main(parse_arguments(argv,r"D:\1pyidentity\imgdata\guest\1314",r"D:\1pyidentity\imgdata\guest\1314.redo"))


