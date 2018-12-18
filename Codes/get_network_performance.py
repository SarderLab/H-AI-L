import numpy as np
import getWsi
from xml_to_mask import xml_to_mask
from joblib import Parallel, delayed
import multiprocessing
from PIL import Image

def get_perf(wsi,xml1,xml2,args):
    if args.wsi_ext != '.tif':
        WSIinfo=getWsi.getWsi(wsi)
        dim_x, dim_y=WSIinfo.dimensions
    else:
        im = Image.open(wsi)
        dim_x, dim_y=im.size

    totalPixels=np.float(dim_x*dim_y)

    # annotated xml
    mask_gt = xml_to_mask(xml1, (0,0), (dim_x,dim_y), 1, 0)
    # predicted xml
    mask_pred = xml_to_mask(xml2, (0,0), (dim_x,dim_y), 1, 0)

    np.place(mask_pred,mask_pred>0,1)
    np.place(mask_gt,mask_gt>0,1)

    TP = float(np.sum(np.multiply(mask_pred, mask_gt)))
    FP = float(np.sum(mask_pred) - TP)

    mask_pred = abs(mask_pred - 1)
    mask_gt = abs(mask_gt - 1)
    np.place(mask_pred,mask_pred>0,1)
    np.place(mask_gt,mask_gt>0,1)

    TN = float(np.sum(np.multiply(mask_pred,mask_gt)))
    FN = float(np.sum(mask_pred) - TN)

    if TP+FP==0:
        precision = 1
    else:
        precision = (TP/(TP+FP))

    accuracy = ((TP + TN) / (TN+FN+TP+FP))

    if TN+FP == 0:
        specificity = 1
    else:
        specificity = (TN/(FP+TN))

    if TP+FN ==0:
        sensitivity= 1
    else:
        sensitivity = (TP / (TP+FN))

    return sensitivity,specificity,precision,accuracy
