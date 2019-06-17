
import numpy as np
from getWsi import getWsi
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, disk, remove_small_objects,label
from scipy.ndimage.morphology import binary_fill_holes
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
from skimage.filters import gaussian
from skimage.morphology import binary_dilation, diamond
def get_choppable_regions(wsi,index_x, index_y, boxSize,white_percent):
    if wsi.split('.')[-1] != 'tif':
        slide=getWsi(wsi)
        slide_level = slide.level_count-1

        fullSize=slide.level_dimensions[0]
        resRatio= 16
        ds_1=fullSize[0]/16
        ds_2=fullSize[1]/16
        Im=np.array(slide.get_thumbnail((ds_1,ds_2)))

        ID=wsi.split('.svs')[0]

        hsv=rgb2hsv(Im)

        g=gaussian(hsv[:,:,1],20)


        binary=(g>0.05).astype('bool')
        binary2=binary_dilation(binary,selem=diamond(20))
        binary2=binary_fill_holes(binary2)

        '''
        Im2=Im
        ax1=plt.subplot(121)
        ax1=plt.imshow(Im)
        ax1=plt.subplot(122)
        Im2[binary2==0,:]=0
        ax1=plt.imshow(Im2)

        plt.savefig(ID+'.png')
        '''

        choppable_regions=np.zeros((len(index_y),len(index_x)))
        for idxy,yi in enumerate(index_y):
            for idxx,xj in enumerate(index_x):
                yStart = int(np.round((yi)/resRatio))
                yStop = int(np.round((yi+boxSize)/resRatio))
                xStart = int(np.round((xj)/resRatio))
                xStop = int(np.round((xj+boxSize)/resRatio))
                box_total=(xStop-xStart)*(yStop-yStart)
                if np.sum(binary2[yStart:yStop,xStart:xStop])>(white_percent*box_total):
                    choppable_regions[idxy,idxx]=1

    else:
        choppable_regions=np.ones((len(index_y),len(index_x)))

    return choppable_regions
