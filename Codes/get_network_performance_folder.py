import numpy as np
import getWsi
from xml_to_mask import xml_to_mask
from joblib import Parallel, delayed
import multiprocessing
from glob import glob
from matplotlib import pyplot as plt
from PIL import Image
#def get_network_performance(WSI_location,xml_annotation,xml_prediction):
block_size=2000
anotDir='/home/bgbl/H-AI-L/IFTAKuang/Annotations/'
predDir='/home/bgbl/H-AI-L/IFTAKuang/TRAINING_data/1/Predicted_XMLs/'
dataDir='/home/bgbl/H-AI-L/IFTAKuang/wsi/'
txtDir='/home/bgbl/H-AI-L/IFTAKuang/'
savelist=[];
f_name1=txtDir + 'performance.txt'
f1=open(f_name1,'w')
f1.close()

def main():
    xmlAnnotation=glob(anotDir + '*.xml')
    xmlPrediction=glob(predDir + '*.xml')
    print(xmlAnnotation)
    for idx,xml in enumerate(xmlAnnotation):

        annotationID=xml.split('/')[-1]
        x1=anotDir + annotationID
        x2=predDir + annotationID
        w=dataDir + annotationID.split('.xml')[0] + '.svs'
        f1=open(f_name1,'a')
        f1.write(str(get_perf(w,x1,x2)) + '\n')
    f1.close()

#
#r=Parallel(n_jobs=num_cores)(delayed(inspect_mask)(yStart=i, xStart=j, xml_annotation=xml_annotation, f_name=f_name, f2_name=f2_name) for i in index_y for j in index_x)
def get_perf(wsi,xml1,xml2,args):
    #specs=inspect_mask(index_y[0],index_x[0],block_size,xml_annotation,xml_prediction)

    if args.wsi_ext != '.tif':
        WSIinfo=getWsi.getWsi(wsi)
        dim_x, dim_y=WSIinfo.dimensions
    else:
        im = Image.open(wsi)
        dim_x, dim_y=im.size

    totalPixels=np.float(dim_x*dim_y)
    index_y=range(0,dim_y-block_size,block_size)
    index_x=range(0,dim_x-block_size,block_size)

    num_cores = multiprocessing.cpu_count()
    r=Parallel(n_jobs=num_cores)(delayed(inspect_mask)(yStart=i, xStart=j, block_size=block_size, annotation_xml=xml1,prediction_xml=xml2) for i in index_y for j in index_x)

    TN=np.zeros((1,5));
    TP=np.zeros((1,5));
    FP=np.zeros((1,5));
    FN=np.zeros((1,5));
    sensitivity=np.zeros((1,5))
    specificity=np.zeros((1,5))
    precision=np.zeros((1,5))

    for classID in range(0,5):
        for t in range(0,len(r)):

            currentspecs=r[t]

            TP[0,classID]=TP[0,classID]+currentspecs[classID,0]
            FP[0,classID]=FP[0,classID]+currentspecs[classID,1]
            FN[0,classID]=FN[0,classID]+currentspecs[classID,2]
            TN[0,classID]=TN[0,classID]+currentspecs[classID,3]

        if (TP[0,classID]+FN[0,classID])==0:
            sensitivity[0,classID]=0
        else:
            sensitivity[0,classID]=np.float(TP[0,classID])/np.float(TP[0,classID]+FN[0,classID])

        if (TN[0,classID]+FP[0,classID])==0:
            specificity[0,classID]=0
        else:
            specificity[0,classID]=np.float(TN[0,classID])/np.float(TN[0,classID]+FP[0,classID])
        #precision[0,classID]=np.float(TP[0,classID])/np.float(TP[0,classID]+FP[0,classID])
    return sensitivity,specificity

def inspect_mask(yStart, xStart,block_size, annotation_xml,prediction_xml): # perform cutting in parallel
    performance=np.zeros((5,4))
    yEnd = yStart+block_size
	#print(yEnd)
    xEnd = xStart+block_size
	#print(xEnd)
    xLen=xEnd-xStart
    yLen=yEnd-yStart
    mask_annotation=xml_to_mask(annotation_xml,[xStart,yStart],[xLen,yLen],1,0)
    prediction_annotation=xml_to_mask(prediction_xml,[xStart,yStart],[xLen,yLen],1,0)
    for classID in range(0,5):
        annotation=mask_annotation==classID
        prediction=prediction_annotation==classID

        TP=(np.sum(np.multiply(annotation,prediction)))
        FP=(np.sum(np.multiply((1-annotation),(prediction))))
        FN=(np.sum(np.multiply((annotation),(1-prediction))))
        TN=(np.sum(np.multiply((1-annotation),(1-prediction))))
        performance[classID,:]=[TP,FP,FN,TN]

    return performance


if __name__ == '__main__':
    main()
