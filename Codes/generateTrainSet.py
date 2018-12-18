import glob
import numpy as np
import os
from cv2 import imread,imwrite


def generateDatalists(images,masks,imfolder,maskfolder,imExt,maskExt,f_name1):
    if os.path.exists(f_name1):
        os.remove(f_name1)
    f1=open(f_name1,'w')
    f1.close()

    trainingNames=glob.glob(images + '*' + imExt)
    totalImages=len(trainingNames)

    f1=open(f_name1,'a')
    for im in range(0,totalImages):
        fileID=trainingNames[im].split('/')[-1].split('.')[0]
        imagename=imfolder + fileID + '.jpeg'
        maskname=maskfolder + fileID + '.png'
        f1.write(imagename + ' ' + maskname + '\n')
    f1.close()
