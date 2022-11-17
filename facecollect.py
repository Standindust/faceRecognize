import sys

from webCam import play
def main(argv):
    play('1920x1080',0,argv)
if __name__ == "__main__":
     # main(argv=[1508,331030200102150018,"guest.guest"])
     main(sys.argv[1:])