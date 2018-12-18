import numpy as np
from scipy.misc import imread
from skimage.color import rgb2hsv,hsv2rgb,rgb2lab,lab2rgb
import matplotlib as plt
import cv2
from skimage import exposure

def randomHSVshift(x,hShift,lShift):
    I=x


    I=rgb2hsv(I)
    I[:,:,0]=(I[:,:,0]+hShift)
    I=hsv2rgb(I)
    I=rgb2lab(I)
    I[:,:,0]=exposure.adjust_gamma(I[:,:,0],lShift)
    I=lab2rgb(I)
    return I
