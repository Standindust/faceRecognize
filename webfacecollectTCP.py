import shutil
import sys
import time
import cv2
import os
import socket, threading
import numpy as np
import requests
from tkinter import Radiobutton, IntVar, Button, Tk, messagebox
confThreshold = 0.9
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
    '240x320': '240x320',
    '320x240': '320x240',
    '352x288': '352x288',
    '480x320': '480x320',
    '480x360': '480x360',
    '480x640': '480x640',
    '640x360': '640x360',
    '640x480': '640x480',
    '640x640': '640x640',
    '720x480': '720x480',
    '864x480': '864x480',
    '1280x640': '1280x640',
    '1280x720': '1280x720',
    '1280x960': '1280x960',
    '1920x960': '1920x960',
    '1920x1080': '1920x1080',
}

tcp_func = {
    # 'Limit_FPS': '/cam/1/fpslimit',
    'Autofocus': b'CMD /v1/ctl?8',
    'Toggle_LED': b'CMD /v1/ctl?9',
    'Zoom_In': b'CMD /v1/ctl?7',
    'Zoom_Out': b'CMD /v1/ctl?6',
    # 'Save_Photo_on_SD': '/cam/1/takepic',        # url
    # 音频流 udp 发送至4748服务端
    'Audio': b'CMD /v2/audio',  # udp 客户端向4748发送,然后获取字节
    'Stop': b'CMD /v1/stop'  # #  udp 客户端向4748发送
}

get_url = {
    'Limit_FPS': '/cam/1/fpslimit',
    'Autofocus': '/cam/1/af',
    'Toggle_LED': '/cam/1/led_toggle',
    'Zoom_In': '/cam/1/zoomin',
    'Zoom_Out': '/cam/1/zoomout',
    'Save_Photo_on_SD': '/cam/1/takepic',
}

s = b'\xff\xd8'
e = b'\xff\xd9'

class DroidCam_Client:
    def __init__(self, master):
        self.master = master
        self.master.protocol("WM_DELETE_WINDOW", self.handler)
        self.playEvent = threading.Event()
        self.size = size_dict['640x640']
        self.v = IntVar()
        self.v.set(8)
        self.createWidgets()
        self.PlatState = False

    def set_size(self):
        self.size = list(size_dict.values())[self.v.get()]

    def createWidgets(self):
        """Build GUI."""

        j = 0
        rr = 0
        cc = -1
        for key in size_dict:
            if cc < 6:
                cc += 1
            else:
                rr += 1
                cc = 0
            Radiobutton(self.master, variable=self.v, text=key, value=j, command=self.set_size).grid(row=rr, column=cc)
            j += 1

        s = len(size_dict) + 1
        self.setup = Button(self.master, width=15, padx=3, pady=3)
        self.setup["text"] = "Play"
        self.setup["command"] = self.Play
        self.setup.grid(row=s, column=0, padx=2, pady=2)

        self.setup = Button(self.master, width=15, padx=3, pady=3)
        self.setup["text"] = "Get_Audio"
        self.setup["command"] = self.Get_Audio
        self.setup.grid(row=s+1, column=0, padx=2, pady=2)

        c = 1
        for func in get_url:
            btn = Button(self.master, width=15, padx=3, pady=3)
            btn["text"] = func
            btn["command"] = getattr(self, func)
            btn.grid(row=s, column=c, padx=2, pady=2)
            c += 1

    def play(self, size, event):
        net = cv2.dnn.readNetFromTensorflow("face_detector/opencv_face_detector_uint8.pb",
                                           "face_detector/opencv_face_detector.pbtxt")

        sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sk.connect(("10.3.171.122", 4747))
        s1 = 'CMD /v2/video{}?'.format(size)
        print(s1)
        sk.send(s1.encode('utf-8'))
        s = sk.recv(1024)
        jpeg_data = b''
        tmp_data = b''
        flag=0

        # type = sys.argv.pop()
        if flag==1:
            type = "guest.guest"
            type1 = type.split('.')[0]
            type2 = type.split('.')[1]
            # idnum = sys.argv.pop()
            idnum = "551030200102150018"
            # room_number = sys.argv.pop()
            room_number = "1103"
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
            while 1:
                msg = sk.recv(1024)
                if e not in msg:
                    jpeg_data += msg
                else:
                    if msg.endswith(e):
                        jpeg_data += msg
                    else:
                        a, b = msg.split(e)
                        jpeg_data = jpeg_data + a + e
                        tmp_data = b


                    frame = bytes2cv(jpeg_data[4:])

                    cv2.rectangle(frame, (180, 100), (480, 400), color=(0, 255, 255), thickness=2)
                    cv2.putText(frame, "Put your face inside rectangle", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
                                3)
                    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False))
                    detections = net.forward()
                    cols = frame.shape[1]
                    rows = frame.shape[0]
                    if num > 400:
                        cv2.putText(frame, "Collection completed", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 3)
                        cv2.imshow("detections", frame)
                        if num == 410:
                            break
                    if num <= 150:
                        cv2.putText(frame, "Turn left slowly", (120, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 3)
                    if num <= 200 and num > 100:
                        cv2.putText(frame, "Turn right slowly", (120, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 3)
                    if num <= 300 and num > 200:
                        cv2.putText(frame, "Lower your head slowly", (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    if num <= 400 and num > 300:
                        cv2.putText(frame, "Raise your head slowly", (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    for i in range(detections.shape[2]):
                        confidence = detections[0, 0, i, 2]

                        if confidence > 0.9:

                            xLeftBottom = int(detections[0, 0, i, 3] * cols)
                            yLeftBottom = int(detections[0, 0, i, 4] * rows)
                            xRightTop = int(detections[0, 0, i, 5] * cols)
                            yRightTop = int(detections[0, 0, i, 6] * rows)

                            if xLeftBottom < 180 or xRightTop > 480 or yLeftBottom < 100 or yRightTop > 400:
                                cv2.putText(frame, "face doesn't match trctangle", (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                            (0, 255, 0), 4)
                                cv2.imshow("detections", frame)

                                break

                            if xRightTop - xLeftBottom > 300 and yRightTop - yLeftBottom > 300:
                                cv2.putText(frame, "stay back", (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 8)
                                cv2.imshow("detections", frame)
                                break

                            if num % 5 == 0:
                                cv2.imencode(".jpg", frame[100:400, 180:480])[1].tofile(
                                    "D:\\1pyidentity\\imgdata\\" + str(type1) + "\\" + str(room_number) + "\\" +
                                    str(room_number) + "." + str(idnum) + '.' + str(num / 5).split('.')[0] +
                                    '.' + type2 + ".jpg")

                                print("成功保存第" + str(num / 5).split('.')[0] + '张照片' + ".jpg")

                            if num <= 150:
                                cv2.putText(frame, str(num / 5) + "/20", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                            (0, 0, 255),
                                            3)

                            if num <= 200 and num > 100:
                                cv2.putText(frame, str(num / 5) + "/40", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                            (0, 0, 255),
                                            3)

                            if num <= 300 and num > 200:
                                cv2.putText(frame, str(num / 5) + "/60", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                            (0, 0, 255),
                                            3)

                            if num <= 400 and num > 300:
                                cv2.putText(frame, str(num / 5) + "/80", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                            (0, 0, 255),
                                            3)

                            cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop), color=(0, 255, 0),
                                          thickness=2)
                            num = num + 1
                        cv2.imshow("detections", frame)

                    if ord(' ') == cv2.waitKey(10):
                        break
                    # if cv.waitKey(0) != -1:
                    #     break
                    self.s = f"{len(frame[0])}x{len(frame)}"
                    # key = cv2.waitKey(1)
                    # if key == 27:
                    #     cv2.destroyWindow(f"DroidCam{size}")
                    #     event.set()
                    #     break
                    jpeg_data = tmp_data
                    tmp_data = b''
            copy()
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
            confThreshold = 0.5
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
            while 1:
                msg = sk.recv(1024)
                if e not in msg:
                    jpeg_data += msg
                else:
                    if msg.endswith(e):
                        jpeg_data += msg
                    else:
                        a, b = msg.split(e)
                        jpeg_data = jpeg_data + a + e
                        tmp_data = b

                    frame = bytes2cv(jpeg_data[4:])
                    if num == 1000:
                        recogizer.read('D:\\1pyidentity\\trainer\\allface\\trainer.yml')
                        print("LOAD")
                        num = 0

                    cols = frame.shape[1]
                    rows = frame.shape[0]

                    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False))
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

                            if xRightTop - xLeftBottom > 300 and yRightTop - yLeftBottom > 300:
                                cv2.putText(frame, "stay back", (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 8)
                                cv2.imshow("detections", frame)
                                break
                            cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop), color=(0, 255, 0),
                                          thickness=2)
                            global confidence1
                            global ids
                            try:
                                ids, confidence1 = recogizer.predict(gray[yLeftBottom:yRightTop, xLeftBottom:xRightTop])
                            except:
                                cv2.putText(frame, "Keep center", (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 8)
                                cv2.imshow("detections", frame)
                                print(str(xLeftBottom) + "\n" + str(xRightTop) + "\n" + str(yLeftBottom) + "\n" + str(
                                    yRightTop))

                            if confidence1 > 45:
                                frame = cv2AddChineseText(frame, "外来人员", (xLeftBottom, yLeftBottom - 20), (0, 255, 0),
                                                          25)

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
                                                          "\nType:" + types[idn.index(alien.get(ids))] + "\nids:" + str(
                                    confidence1),

                                                          (xLeftBottom, yLeftBottom - 80), (0, 255, 0), 25)

                    cv2.imshow("detections", frame)
                    if ord(' ') == cv2.waitKey(10):
                        break
                    num = num + 1
                    self.s = f"{len(frame[0])}x{len(frame)}"
                    jpeg_data = tmp_data
                    tmp_data = b''
            cv2.destroyAllWindows()







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
        ret = requests.get('http://10.3.171.122:4747/cam/1/led_toggle')

    def Limit_FPS(self):
        ret = requests.get('http://10.3.171.122:4747/cam/1/fpslimit')

    def Autofocus(self):
        ret = requests.get('http://10.3.171.122:4747/cam/1/af')

    def Zoom_In(self):
        ret = requests.get('http://10.3.171.122:4747/cam/1/zoomin')

    def Zoom_Out(self):
        ret = requests.get('http://10.3.171.122:4747/cam/1/zoomout')

    def OverRide(self):
        ret = requests.get('http://10.3.171.122:4747/override')

    def Save_Photo_on_SD(self):
        ret = requests.get('http://10.3.171.122:4747/cam/1/takepic')

    def Get_Audio(self):
        f = open("123.mp3","wb")
        sk = socket.socket(type=socket.SOCK_DGRAM)
        ipaddr = ("192.168.124.3", 4748)
        sk.sendto(b"CMD /v2/audio", ipaddr)
        count = 0
        while 1:
            count += 1
            msg, _ = sk.recvfrom(1024)
            if msg:
                # 解析音频数据
                print(count)
                f.write(msg[1:])
            else:
                break
            if count == 10000: break

        print("exit")
        f.flush()
        f.close()
        sk.close()

def run():
    root = Tk()
    app = DroidCam_Client(root)
    app.master.title("DroidCam_TCP_Client")
    root.mainloop()

if __name__ == '__main__':
    run()

