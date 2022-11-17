import cv2
import os
import numpy as np
from skimage import transform as trans
from PIL import Image
# test_fddb.py这个脚本，见我的整理的项目地址。位于insightface人脸识别代码记录(总)(基于MXNet)中。
from test_fddb import detect


# 这里是一个简单的由路径读取图片的方法
def read_image(img_path, **kwargs):
    mode = kwargs.get('mode', 'rgb')
    layout = kwargs.get('layout', 'HWC')
    if mode == 'gray':
        img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    else:
        img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_COLOR)
        if mode == 'rgb':
            # print('to rgb')
            img = img[..., ::-1]
        if layout == 'CHW':
            img = np.transpose(img, (2, 0, 1))
    return img


# 这个方法是核心。就是根据检测
def preprocess(img, bbox_list=None, landmark_list=None, **kwargs):
    # 这是存放校正后图像的一个列表
    warped_list = list()
    # 一张图片上可能存在多个人脸，即有多个bbox和关键点信息，需要分别进行裁剪和校正
    for bbox, landmark in zip(bbox_list, landmark_list):

        if isinstance(img, str):
            img = read_image(img, bbox, **kwargs)

        M = None
        image_size = []
        # 对裁剪后图像大小的控制112*112
        str_image_size = kwargs.get('image_size', '')
        if len(str_image_size) > 0:
            image_size = [int(x) for x in str_image_size.split(',')]
            if len(image_size) == 1:
                image_size = [image_size[0], image_size[0]]
            assert len(image_size) == 2
            assert image_size[0] == 112
            assert image_size[0] == 112 or image_size[1] == 96
        # src存放着标准关键点信息所在的位置
        if landmark is not None:
            assert len(image_size) == 2
            src = np.array([
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041]], dtype=np.float32)
            # 这里这个+8么太明白是什么原理...(有知道的小伙伴希望可以评论留言)
            if image_size[1] == 112:
                src[:, 0] += 8.0
            dst = landmark.astype(np.float32)
            # 创建相似变换矩阵
            tform = trans.SimilarityTransform()
            # 根据标准关键landmark和所得到的landmark来计算得到变换矩阵
            tform.estimate(dst, src)
            # 提供给后面仿射变换函数使用。因为仿射变换只涉及旋转和平移，所以只需要矩阵中的前两个参数
            # dst(x,y)=src(M11x+M12x+M13,M21x+M22y+M23)
            M = tform.params[0:2, :]
        # 这个是变换矩阵不存在的情况下，暂时不做讨论
        if M is None:
            if bbox is None:  # use center crop
                det = np.zeros(4, dtype=np.int32)
                det[0] = int(img.shape[1] * 0.0625)
                det[1] = int(img.shape[0] * 0.0625)
                det[2] = img.shape[1] - det[0]
                det[3] = img.shape[0] - det[1]
            else:
                det = bbox
            margin = kwargs.get('margin', 44)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img.shape[1])
            bb[3] = np.minimum(det[3] + margin / 2, img.shape[0])
            ret = img[bb[1]:bb[3], bb[0]:bb[2], :]
            if len(image_size) > 0:
                ret = cv2.resize(ret, (image_size[1], image_size[0]))
            return ret
        else:  # do align using landmark
            assert len(image_size) == 2
            # 然后调用opencv中的放射变换得到校正后的人脸
            warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)
            warped_list.append(warped)

    return warped_list


def Align(img_path):
    # 调用retinaface中的人脸检测器，得到bbox和landmark坐标信息
    _ret = detect(img_path)
    # 因为上述得到的一个list，list[0]存放了人脸的信息，list[1]是其所在路径
    ret = _ret[0]

    # 将人脸置信度抽取出来，进行一个筛选。因为在检测器中进行过sort，得到的
    # 置信度高的人脸位于前面，所以当出现第一个小于阈值的情况，后面便均小于。
    scores = ret[:, 4]
    index = list()
    for i in range(len(scores)):
        if scores[i] > 0.5:
            index.append(i)
        else:
            break

    bbox = ret[index, 0:4]
    points_ = ret[index, 5:15]
    points = points_.copy()
    # 因为得到的landmark是(x1,y1,x2,y2....)这种形式，需要变换成(x1,x2,...y1,y2....)这种形式
    points[index, 0:5:1] = points_[index, 0::2]
    points[index, 5:10:1] = points_[index, 1::2]

    if bbox.shape[0] == 0:
        return None
    # 将上述得到的bbox和landmark信息分别存放到如下list
    points_list = list()
    bbox_list = list()
    for i in range(len(index)):
        point = points[i, :].reshape((2, 5)).T
        bbox_ = bbox[i]
        points_list.append(point)
        bbox_list.append(bbox_)
    # points = points.reshape((len(index),2,5)).transpose(0,2,1)

    face_img = cv2.imread(img_path)
    # 调用此方法，得到裁剪和校正过的人脸
    nimg_list = preprocess(face_img, bbox_list, points_list, image_size='112,112')
    # 一张图片存在多张人脸，分别裁剪校正
    aligned_list = list()
    for _, nimg in enumerate(nimg_list):
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(nimg, (2, 0, 1))
        aligned_list.append(aligned)
    return aligned_list


if __name__ == "__main__":
    # path = './data/widerface/train/images/7--Cheering/7_Cheering_Cheering_7_321.jpg'  #单人脸
    # path1 = './data/widerface/train/images/4--Dancing/4_Dancing_Dancing_4_317.jpg'   #多人脸
    path = 'F:\Project\insightface-master\deploy\img'

    # os.walk()返回三个参数:
    #	①root,所在路径,即输入path
    #	②dirs,所在路径下的文件夹
    #	③files,所在路径下的文件
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            img_root = os.path.join(root, dir)
            images = os.listdir(img_root)
            for image in images:
                path = os.path.join(img_root, image)
                print(path)
                out_list = Align(path)
                for i, out in enumerate(out_list):
                    new_image = np.transpose(out, (1, 2, 0))[:, :, ::-1]
                    out = Image.fromarray(new_image)
                    out = out.resize((112, 112))
                    out = np.asarray(out)
                    # 对所得到的人脸进行和原所在文件夹和命名的区别
                    if not os.path.exists(os.path.join(root, '_' + dir)):
                        os.mkdir(os.path.join(root, '_' + dir))
                    # 采用原图像名字和i命名，i表示这张图片中的第几个人脸
                cv2.imwrite(os.path.join(root, '_' + dir, str(image[0:-4]) + '_' + str(i) + '.jpg'), out)
                print('over!')
        # cv2.imshow("out",out)
        # cv2.waitKey()
