import sys

from webCam import play
def main(argv):
    play('1920x1080', 2, argv)

if __name__ == '__main__':
     # main(argv=["1108","guest"])
     main(sys.argv[1:])
