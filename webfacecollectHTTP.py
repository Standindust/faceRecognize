#!/usr/bin/bin python3
# -*- coding: utf-8 -*-
# @Time    : 2020/8/1 19:42
# @Author  : SuperDeng
# @Email   : 1821333144@qq.com
# @File    : socket_client.py
# 基于http协议传输的视频数据流播放
import shutil
import sys
import time
import cv2
import socket, threading
import numpy as np
import requests
from tkinter import Radiobutton, IntVar, Button, Tk, messagebox
import os
from PIL import Image, ImageDraw, ImageFont

def bytes2cv(im):
    '''二进制图片转cv2

    :param im: 二进制图片数据，bytes
    :return: cv2图像，numpy.ndarray
    '''
    return cv2.imdecode(np.array(bytearray(im), dtype='uint8'), cv2.IMREAD_UNCHANGED)  # 从二进制图片数据中读取

def cv2bytes(im):
    '''cv2转二进制图片

    :param im: cv2图像，numpy.ndarray
    :return: 二进制图片数据，bytes
    '''
    return np.array(cv2.imencode('.png', im)[1]).tobytes()

size_dict = {
    # '240p': '320x240',
    # '480p': '640x480',
    # '720p': '960x720',
    # 'FHD 720p': '1280x720',
    'FHD 1080p': '1920x1080',

}

get_url = {
    'Limit_FPS': '/cam/1/fpslimit',
    'Autofocus': '/cam/1/af',
    'Toggle_LED': '/cam/1/led_toggle',
    'Zoom_In': '/cam/1/zoomin',
    'Zoom_Out': '/cam/1/zoomout',
    'Save_Photo_on_SD': '/cam/1/takepic',
}

class DroidCam_Client:
    def __init__(self, master):
        self.master = master
        self.master.protocol("WM_DELETE_WINDOW", self.handler)
        self.playEvent = threading.Event()
        self.size = size_dict['FHD 1080p']
        self.v = IntVar()
        self.v.set(1)
        self.createWidgets()
        self.PlatState = False

    def set_size(self):
        self.size = list(size_dict.values())[self.v.get()]

    def createWidgets(self):
        """Build GUI."""

        j = 0
        for key in size_dict:
            Radiobutton(self.master, variable=self.v, text=key, value=j, command=self.set_size).grid()
            j += 1

        self.setup = Button(self.master, width=15, padx=3, pady=3)
        self.setup["text"] = "Play"
        self.setup["command"] = self.Play
        self.setup.grid(row=5, column=0, padx=2, pady=2)

        c = 1
        for func in get_url:
            btn = Button(self.master, width=15, padx=3, pady=3)
            btn["text"] = func
            btn["command"] = getattr(self, func)
            btn.grid(row=5, column=c, padx=2, pady=2)
            c += 1

    def play(self, size, event):
        print(size)
        sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sk.connect(("10.3.173.85", 4747))
        data = 'GET /mjpegfeed?{} HTTP/1.1\r\nHost: 10.3.173.85:4747\r\nUser-Agent: Mozilla/5.0 (Windows NT 10.0; WOW64; rv:53.0) Gecko/20100101 Firefox/53.0\r\nAccept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\nAccept-Language: zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3\r\nAccept-Encoding: gzip, deflate\r\nConnection: keep-alive\r\nUpgrade-Insecure-Requests: 1\r\n\r\n'.format(
            size)
        sk.send(data.encode('utf-8'))
        msg = sk.recv(1024)
        recv_dict = msg.decode('utf-8').split('\r\n')
        if 'Connection: Keep-Alive' in recv_dict:
            self.__play(sk,size,event)
        else:
            print('DroidCam is connected to another client.')
            print('Disconnect the other client and Take Over')
            sk.close()

    def __play(self,sk,size,event):
        JPEG_header = b''
        JPEG = b''
        net = cv2.dnn.readNetFromTensorflow("face_detector/opencv_face_detector_uint8.pb",
                                            "face_detector/opencv_face_detector.pbtxt")
        flag=0
        if flag==1:
            type = "guest.guest"
            type1 = type.split('.')[0]
            type2 = type.split('.')[1]
            # idnum = sys.argv.pop()
            idnum = "331030200102150018"
            # room_number = sys.argv.pop()
            room_number = "1101"
            print('\n 正在打开摄像头。。。。。。。')
            # 录入人员的标签，每个人的标签不能相同
            des = r"D:\\1pyidentity\\imgdata\\allface"  # 目标文件夹路径
            if (os.path.exists(des) == 0):
                os.mkdir(r"D:\\1pyidentity\\imgdata\\allface")

            # 捕获摄像头图像
            path = r"D:\1pyidentity\imgdata\\" + type1 + r"\\" + str(room_number)
            if (os.path.exists(path) == 0):
                os.mkdir(r"D:\1pyidentity\imgdata\\" + type1 + r"\\" + str(room_number))

            def copy():
                print("copy")
                for file in os.listdir(path):
                    # 遍历原文件夹中的文件
                    full_file_name = os.path.join(path, file)  # 把文件的完整路径得到
                    # print("要被复制的全文件路径全名:", full_file_name)
                    if os.path.isfile(full_file_name):  # 用于判断某一对象(需提供绝对路径)是否为文件
                        shutil.copy(full_file_name, des)  # shutil.copy函数放入原文件的路径文件全名  然后放入目标文件夹
                print("copy finish")

            num = 1
            while True:
                msg = sk.recv(1024)
                if msg:
                    if b'\r\n\r\n' in msg:
                        # 分界点到了
                        JPEG_header += msg.split(b'\r\n\r\n')[0]
                        long = int(JPEG_header.split(b'\r\n')[-1].split(b':')[-1])
                        JPEG_header = b''
                        JPEG += msg.split(b'\r\n\r\n')[1]
                        while True:
                            JPEG += sk.recv(1024)
                            if len(JPEG) >= long:
                                frame = bytes2cv(JPEG[:long])

                                # 1920 1080
                                cv2.rectangle(frame, (560, 140), (1360, 940), color=(0, 255, 255), thickness=2)
                                cv2.putText(frame, "Put your face inside rectangle", (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                            (0, 0, 255),
                                            3)
                                net.setInput(
                                    cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False))
                                detections = net.forward()
                                cols = frame.shape[1]
                                rows = frame.shape[0]
                                if num > 200:
                                    cv2.putText(frame, "Collection completed", (660, 640), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                                (0, 0, 255), 3)
                                    cv2.imshow("detections", frame)
                                    if num == 210:
                                        break
                                if num <= 50:
                                    cv2.putText(frame, "Turn left slowly", (700, 1020), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                                (0, 0, 255), 3)
                                if num <= 100 and num > 50:
                                    cv2.putText(frame, "Turn right slowly", (700, 1020), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                                (0, 0, 255), 3)
                                if num <= 150 and num > 100:
                                    cv2.putText(frame, "Lower your head slowly", (600, 1020), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                                (0, 0, 255), 3)
                                if num <= 200 and num > 150:
                                    cv2.putText(frame, "Raise your head slowly", (600, 1020), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                                (0, 0, 255), 3)
                                for i in range(detections.shape[2]):
                                    confidence = detections[0, 0, i, 2]

                                    if confidence > 0.9:

                                        xLeftBottom = int(detections[0, 0, i, 3] * cols)
                                        yLeftBottom = int(detections[0, 0, i, 4] * rows)
                                        xRightTop = int(detections[0, 0, i, 5] * cols)
                                        yRightTop = int(detections[0, 0, i, 6] * rows)
                                        # (660, 240), (1260, 840)
                                        if xLeftBottom < 660 or xRightTop > 1360 or yLeftBottom < 140 or yRightTop > 940:
                                            cv2.putText(frame, "face doesn't match trctangle", (500, 450),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 2,
                                                        (0, 255, 0), 4)
                                            cv2.imshow("detections", frame)

                                            break

                                        if xRightTop - xLeftBottom > 800 and yRightTop - yLeftBottom > 800:
                                            cv2.putText(frame, "stay back", (500, 200), cv2.FONT_HERSHEY_SIMPLEX, 4,
                                                        (0, 255, 0), 8)
                                            cv2.imshow("detections", frame)
                                            break

                                        if num % 5 == 0:
                                            cv2.imencode(".jpg", frame[140:940,660:1360])[1].tofile(
                                                "D:\\1pyidentity\\imgdata\\" + str(type1) + "\\" + str(room_number) + "\\" +
                                                str(room_number) + "." + str(idnum) + '.' + str(num / 5).split('.')[0] +
                                                '.' + type2 + ".jpg")

                                            print("成功保存第" + str(num / 5).split('.')[0] + '张照片' + ".jpg")


                                        if num <= 200:
                                            cv2.putText(frame, str(num / 5) + "/40", (0, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                                        2,
                                                        (0, 0, 255),
                                                        3)

                                        cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                                                      color=(0, 255, 0),
                                                      thickness=2)
                                        num = num + 1

                                # # 显示
                                cv2.namedWindow("detections", 0)
                                cv2.resizeWindow("detections", 1280 , 720)  # 自己设定窗口图片的大小
                                cv2.imshow("detections", frame)



                                if ord(' ') == cv2.waitKey(10):

                                    break




                                # cv2.imshow(f"DroidCam{size}", frame)
                                # cv2.waitKey(1)
                                JPEG_header += JPEG[long:]
                                JPEG = b''
                                break
                    else:
                        JPEG_header += msg
                    if event.isSet():
                        cv2.destroyWindow(f"DroidCam{size}")

                        break
                if num==210:
                    break
            copy()
            sk.close()


        if flag==0:
            recogizer = cv2.face.LBPHFaceRecognizer_create()
            # 读取训练好的系统文件

            # 存储人脸库中人员的名字
            roomnumbers = []
            # 对应的标签
            idn = []

            alien = {}

            types = []
            inWidth = 300
            inHeight = 300
            confThreshold = 0.9
            count = 0
            def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
                if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
                    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB
                                                      ))
                # 创建一个可以在给定图像上绘图的对象
                draw = ImageDraw.Draw(img)
                # 字体的格式
                fontStyle = ImageFont.truetype(
                    "simsun.ttc", textSize, encoding="utf-8")
                # 绘制文本
                draw.text(position, text, textColor, font=fontStyle)
                # 转换回OpenCV格式
                return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

            def name():
                path = 'D:\\1pyidentity\\imgdata\\allface'
                imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
                for imagePath in imagePaths:
                    id = str(os.path.split(imagePath)[1].split('.', 3)[1])
                    id1 = int(id[0] + id[5] + id[6] + id[9] + id[11] + id[13] + id[15])

                    if alien.get(id) == None:
                        alien[id1] = id

                    roomnumber = int(os.path.split(imagePath)[1].split('.', 3)[0])
                    type = str(os.path.split(imagePath)[1].split('.')[3])
                    roomnumbers.append(roomnumber)
                    idn.append(id)
                    types.append(type)

            name()
            num = 0
            recogizer.read('D:\\1pyidentity\\trainer\\allface\\trainer.yml')
            while True:
                msg = sk.recv(1024)
                if msg:
                    if b'\r\n\r\n' in msg:
                        # 分界点到了
                        JPEG_header += msg.split(b'\r\n\r\n')[0]
                        long = int(JPEG_header.split(b'\r\n')[-1].split(b':')[-1])
                        JPEG_header = b''
                        JPEG += msg.split(b'\r\n\r\n')[1]
                        while True:
                            JPEG += sk.recv(1024)
                            if len(JPEG) >= long:
                                frame = bytes2cv(JPEG[:long])

                                if num == 1000:
                                    recogizer.read('D:\\1pyidentity\\trainer\\allface\\trainer.yml')
                                    print("LOAD")
                                    num = 0

                                cols = frame.shape[1]
                                rows = frame.shape[0]

                                net.setInput(
                                    cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False))
                                detections = net.forward()

                                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                                for i in range(detections.shape[2]):
                                    confidence = detections[0, 0, i, 2]

                                    if confidence > confThreshold:

                                        xLeftBottom = int(detections[0, 0, i, 3] * cols)
                                        yLeftBottom = int(detections[0, 0, i, 4] * rows)
                                        xRightTop = int(detections[0, 0, i, 5] * cols)
                                        yRightTop = int(detections[0, 0, i, 6] * rows)
                                        if xLeftBottom < 0: xLeftBottom = 1
                                        if xRightTop < 0: xRightTop = 1
                                        if yLeftBottom < 0: xLeftBottom = 1
                                        if yRightTop < 0: yRightTop = 1

                                        if xRightTop - xLeftBottom > 800 and yRightTop - yLeftBottom > 800:
                                            cv2.putText(frame, "stay back", (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 4,
                                                        (0, 255, 0), 8)

                                            break
                                        cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                                                      color=(0, 255, 0),
                                                      thickness=2)
                                        global confidence1
                                        global ids
                                        try:
                                            ids, confidence1 = recogizer.predict(
                                                gray[yLeftBottom:yRightTop, xLeftBottom:xRightTop])
                                        except:
                                            cv2.putText(frame, "Keep in center", (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 4,
                                                        (0, 255, 0), 8)
                                            cv2.imshow("Face recognition", frame)
                                            print(str(xLeftBottom) + "\n" + str(xRightTop) + "\n" + str(
                                                yLeftBottom) + "\n" + str(
                                                yRightTop))

                                        if confidence1 > 40:
                                            frame = cv2AddChineseText(frame, "外来人员", (xLeftBottom, yLeftBottom - 20),
                                                                      (0, 255, 0),
                                                                      50)

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

                                            frame = cv2AddChineseText(frame, title + str(
                                                roomnumbers[idn.index(alien.get(ids))]) +
                                                                      "\nID:" + alien.get(ids) +
                                                                      "\nType:" + types[
                                                                          idn.index(alien.get(ids))] + "\nids:" + str(
                                                confidence1),

                                                                      (xLeftBottom, yLeftBottom - 80), (0, 255, 0), 50)
                                cv2.namedWindow("Face recognition", 0)
                                cv2.resizeWindow("Face recognition", 1280, 720)
                                cv2.imshow("Face recognition", frame)
                                if ord(' ') == cv2.waitKey(10):
                                    break
                                num = num + 1
                                # cv2.imshow(f"DroidCam{size}", frame)
                                # cv2.waitKey(1)
                                JPEG_header += JPEG[long:]
                                JPEG = b''
                                break
                    else:
                        JPEG_header += msg
                    if event.isSet():
                        cv2.destroyAllWindows()

                        break
            sk.close()

    def Play(self):
        if not self.PlatState:
            threading.Thread(target=self.play, args=(self.size, self.playEvent)).start()
            self.PlatState = True
        else:
            self.playEvent.set()
            self.OverRide()
            self.playEvent.clear()
            time.sleep(1)
            threading.Thread(target=self.play, args=(self.size, self.playEvent)).start()
            self.PlatState = True

    def handler(self):
        """Handler on explicitly closing the GUI window."""
        if messagebox.askokcancel("Quit?", "Are you sure you want to quit?"):
            self.playEvent.set()
            time.sleep(1)
            self.master.destroy()  # Close the gui window
            sys.exit(0)
        else:  # When the user presses cancel, resume playing.
            return

    def Toggle_LED(self):
        ret = requests.get('http://10.3.173.85:4747/cam/1/led_toggle')

    def Limit_FPS(self):
        ret = requests.get('http://10.3.173.85:4747/cam/1/fpslimit')

    def Autofocus(self):
        ret = requests.get('http://10.3.173.85:4747/cam/1/af')

    def Zoom_In(self):
        ret = requests.get('http://10.3.173.85:4747/cam/1/zoomin')

    def Zoom_Out(self):
        ret = requests.get('http://10.3.173.85:4747/cam/1/zoomout')

    def OverRide(self):
        ret = requests.get('http://10.3.173.85:4747/override')

    def Save_Photo_on_SD(self):
        ret = requests.get('http://10.3.173.85:4747/cam/1/takepic')

def run():
    root = Tk()
    app = DroidCam_Client(root)
    app.master.title("DroidCam_Client")
    root.mainloop()

if __name__ == '__main__':
    run()