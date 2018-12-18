import glob
import cv2




predictionAddress='/hdd/wsi_fun/Codes/Deeplab-v2--ResNet-101/outputAugmentTest/prediction/*.png'
imageAddress='/hdd/wsi_fun/ImageAugCustom/AugmentationOutput/Images/'
maskList=glob.glob(predictionAddress)
Total=len(maskList)


for im in range(0,Total):
    fileID=maskList[im].split('/')
    fileID=fileID[-1]
    fileID=fileID.split('_mask')
    fileID=fileID[0]
    if 'K14' in fileID:

        mask=(cv2.imread(maskList[im],0)+1)*.5
        image=cv2.imread(imageAddress + fileID + '.jpeg',1)
        image[:,:,0]=image[:,:,0]*(mask)
        image[:,:,1]=image[:,:,1]*(mask)
        image[:,:,2]=image[:,:,2]*(mask)
        cv2.imshow('image',image)
        cv2.waitKey(500)
    elif 'K17' in fileID:

        mask=(cv2.imread(maskList[im],0)+1)*.5
        image=cv2.imread(imageAddress + fileID + '.jpeg',1)
        image[:,:,0]=image[:,:,0]*(mask)
        image[:,:,1]=image[:,:,1]*(mask)
        image[:,:,2]=image[:,:,2]*(mask)
        cv2.imshow('image',image)
        cv2.waitKey(500)
    elif 'K13' in fileID:

        mask=(cv2.imread(maskList[im],0)+1)*.5
        image=cv2.imread(imageAddress + fileID + '.jpeg',1)
        image[:,:,0]=image[:,:,0]*(mask)
        image[:,:,1]=image[:,:,1]*(mask)
        image[:,:,2]=image[:,:,2]*(mask)
        cv2.imshow('image',image)
        cv2.waitKey(500)
    elif 'K16' in fileID:

        mask=(cv2.imread(maskList[im],0)+1)*.5
        image=cv2.imread(imageAddress + fileID + '.jpeg',1)
        image[:,:,0]=image[:,:,0]*(mask)
        image[:,:,1]=image[:,:,1]*(mask)
        image[:,:,2]=image[:,:,2]*(mask)
        cv2.imshow('image',image)
        cv2.waitKey(500)
    else:
        continue
