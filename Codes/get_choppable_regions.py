
import numpy as np
from getWsi import getWsi
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, disk, remove_small_objects,label
from scipy.ndimage.morphology import binary_fill_holes

def get_choppable_regions(wsi,index_x, index_y, boxSize):
    if wsi.split('.')[-1] != 'tif':
        slide=getWsi(wsi)
        slide_level = slide.level_count-1
        thumbSize=slide.level_dimensions[slide_level]
        fullSize=slide.level_dimensions[0]
        resRatio= (fullSize[0]/thumbSize[0])
        Im=np.array(slide.read_region((0,0),slide_level,(thumbSize)))
        ID=wsi.split('.svs')[0]

        grayImage=0.2125*Im[:,:,0]+0.7154*Im[:,:,1]+0.0721*Im[:,:,2]
        grayImage[grayImage == 0] = 255

        #thresh = threshold_otsu(grayImage)
        thresh = 240
        binary =grayImage<thresh
        binary=binary_fill_holes(binary)
        #binary=binary(binary,se)
        #binary=binary_fill_holes(binary)
        #binary=remove_small_objects(binary,object_size)

        choppable_regions=np.zeros((len(index_y),len(index_x)))
        for idxy,yi in enumerate(index_y):
            for idxx,xj in enumerate(index_x):
                yStart = int(np.round((yi)/resRatio))
                yStop = int(np.round((yi+boxSize)/resRatio))
                xStart = int(np.round((xj)/resRatio))
                xStop = int(np.round((xj+boxSize)/resRatio))
                if np.sum(binary[yStart:yStop,xStart:xStop])>0:
                    choppable_regions[idxy,idxx]=1

    else:
        choppable_regions=np.ones((len(index_y),len(index_x)))

    return choppable_regions
