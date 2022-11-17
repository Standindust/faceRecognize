import  os
i=1
path=r"D:\1pyidentity\imgdata\guest\1514"
imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
for imagePath in imagePaths:

    room =os.path.split(imagePath)[1].split('.')
    str1 = "1514.151030200102150018."+str(i)+".guest.jpg"

    os.rename(imagePath,path+r"\\"+str1)
    i=i+1