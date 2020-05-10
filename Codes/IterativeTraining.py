import numpy as np
import multiprocessing
import os
import sys
import cv2
import matplotlib.pyplot as plt
import time
import random
import warnings
import argparse

from skimage.transform import resize
from skimage.io import imread, imsave
from skimage.morphology import remove_small_objects
from skimage.color import rgb2lab
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import binary_fill_holes
from glob import glob
from getWsi import getWsi
from xml_to_mask import xml_to_mask,get_num_classes
from joblib import Parallel, delayed
from shutil import rmtree,move,copyfile
from imgaug import augmenters as iaa
from randomHSVshift import randomHSVshift
from generateTrainSet import generateDatalists
from subprocess import call
from get_choppable_regions import get_choppable_regions
"""

Code for - cutting / augmenting / training CNN

This uses WSI and XML files to train 2 neural networks for semantic segmentation
    of histopath tissue via human in the loop training

"""

global seq #Define geometric augmentation strategies
seq=iaa.Sequential([
iaa.Fliplr(0.5),
iaa.Flipud(0.5),
iaa.PiecewiseAffine(scale=(0.01, 0.05),order=0),
])

#Record start time
totalStart=time.time()

def IterateTraining(args):
    ## calculate low resolution block params
    downsampleLR = int(args.downsampleRateLR**.5) #down sample for each dimension
    region_sizeLR = int(args.boxSizeLR*(downsampleLR)) #Region size before downsampling
    stepLR = int(region_sizeLR*(1-args.overlap_percentLR)) #Step size before downsampling
    ## calculate low resolution block params
    downsampleHR = int(args.downsampleRateHR**.5) #down sample for each dimension
    region_sizeHR = int(args.boxSizeHR*(downsampleHR)) #Region size before downsampling
    stepHR = int(region_sizeHR*(1-args.overlap_percentHR)) #Step size before downsampling


    global classEnumLR,classEnumHR
    dirs = {'imExt': '.jpeg'}
    dirs['basedir'] = args.base_dir
    dirs['maskExt'] = '.png'
    dirs['modeldir'] = '/MODELS/'
    dirs['tempdirLR'] = '/TempLR/'
    dirs['tempdirHR'] = '/TempHR/'
    dirs['pretraindir'] = '/Deeplab_network/'
    dirs['training_data_dir'] = '/TRAINING_data/'
    dirs['model_init'] = 'deeplab_resnet.ckpt'
    dirs['project']= '/' + args.project
    dirs['data_dir_HR'] = args.base_dir +'/' + args.project + '/Permanent/HR/'
    dirs['data_dir_LR'] = args.base_dir +'/' +args.project + '/Permanent/LR/'


    ##All folders created, initiate WSI loading by human
    #raw_input('Please place WSIs in ')

    ##Check iteration session

    currentmodels=os.listdir(dirs['basedir'] + dirs['project'] + dirs['modeldir'])

    currentAnnotationIteration=check_model_generation(dirs)

    print('Current training session is: ' + str(currentAnnotationIteration))

    ##Create objects for storing class distributions
    annotatedXMLs=glob(dirs['basedir'] + dirs['project'] + dirs['training_data_dir'] + str(currentAnnotationIteration) + '/*.xml')
    classes=[]
    if args.classNum == 0:
        for xml in annotatedXMLs:
            classes.append(get_num_classes(xml))

        classNum_LR = max(classes)
        classNum_HR = max(classes)
    else:
        classNum_LR = args.classNum
        if args.classNum_HR != 0:
            classNum_HR = args.classNum_HR
        else:
            classNum_HR = classNum_LR
    classEnumLR=np.zeros([classNum_LR,1])
    classEnumHR=np.zeros([classNum_HR,1])


    ##for all WSIs in the initiating directory:
    if args.chop_data == 'True':
        print('Chopping')

        start=time.time()
        for xmlID in annotatedXMLs:

            #Get unique name of WSI
            fileID=xmlID.split('/')[-1].split('.xml')[0]

            #create memory addresses for wsi files
            for ext in [args.wsi_ext]:
                wsiID=dirs['basedir'] + dirs['project']+  dirs['training_data_dir'] + str(currentAnnotationIteration) +'/'+ fileID + ext

                #Ensure annotations exist
                if os.path.isfile(wsiID)==True:
                    break

            #Ensure annotations exist
            if os.path.isfile(wsiID)==False:
                print('\nError - missing wsi file: ' + wsiID + ' Please provide.\n')

            #Load openslide information about WSI
            if ext != '.tif':
                slide=getWsi(wsiID)
                #WSI level 0 dimensions (largest size)
                dim_x,dim_y=slide.dimensions
            else:
                im = Image.open(wsiID)
                dim_x, dim_y=im.size
            wsi_mask=xml_to_mask(xmlID, [0,0], [dim_x,dim_y])
            print('Loaded mask')

            #Generate iterators for parallel chopping of WSIs in low resolution
            index_yLR=np.array(range(0,dim_y,stepLR))
            index_xLR=np.array(range(0,dim_x,stepLR))

            index_yLR[-1]=dim_y-stepLR
            index_xLR[-1]=dim_x-stepLR
            #Create memory address for chopped images low resolution
            outdirLR=dirs['basedir'] + dirs['project'] + dirs['tempdirLR']

            #Enumerate cpu core count
            num_cores = multiprocessing.cpu_count()

            #Perform low resolution chopping in parallel and return the number of
            #images in each of the labeled classes
            chop_regions=get_choppable_regions(wsi=wsiID,
                index_x=index_xLR,index_y=index_yLR,boxSize=region_sizeLR,white_percent=args.white_percent)


            classEnumCLR=Parallel(n_jobs=num_cores)(delayed(return_region)(args=args,
                wsi_mask=wsi_mask, wsiID=wsiID,
                fileID=fileID, yStart=j, xStart=i, idxy=idxy,
                idxx=idxx, downsampleRate=args.downsampleRateLR,
                outdirT=outdirLR, region_size=region_sizeLR,
                dirs=dirs, chop_regions=chop_regions,classNum=classNum_LR) for idxx,i in enumerate(index_xLR) for idxy,j in enumerate(index_yLR))
            print('Time for low res WSI chopping: ' + str(time.time()-start))

            #Add number of images in each class to the global count low resolution
            CSLR=(sum(classEnumCLR))
            for c in range(0,CSLR.shape[0]):
                classEnumLR[c]=classEnumLR[c]+CSLR[c]

            #Print enumerations for each class

            #Generate iterators for parallel chopping of WSIs in high resolution
            index_yHR=np.array(range(0,dim_y,stepHR))
            index_xHR=np.array(range(0,dim_x,stepHR))
            index_yHR[-1]=dim_y-stepHR
            index_xHR[-1]=dim_x-stepHR
            #Create memory address for chopped images high resolution
            outdirHR=dirs['basedir'] + dirs['project'] + dirs['tempdirHR']

            #Perform high resolution chopping in parallel and return the number of
            #images in each of the labeled classes
            chop_regions=get_choppable_regions(wsi=wsiID,
                index_x=index_xHR,index_y=index_yHR,boxSize=region_sizeHR,white_percent=args.white_percent)

            classEnumCHR=Parallel(n_jobs=num_cores)(delayed(return_region)(args=args,
                wsi_mask=wsi_mask, wsiID=wsiID,
                fileID=fileID, yStart=j, xStart=i, idxy=idxy,
                idxx=idxx, downsampleRate=args.downsampleRateHR,
                outdirT=outdirHR, region_size=region_sizeHR,
                dirs=dirs, chop_regions=chop_regions,classNum=classNum_HR) for idxx,i in enumerate(index_xHR) for idxy,j in enumerate(index_yHR))
            print('Time for high res WSI chopping: ' + str(time.time()-start))

            #Add number of images in each class to the global count high resolution
            CSHR=(sum(classEnumCHR))
            for c in range(0,CSHR.shape[0]):
                classEnumHR[c]=classEnumHR[c]+CSHR[c]

            #classEnumHR=[float(6334),float(488)]
            #Print enumerations for each class

        print('Time for WSI chopping: ' + str(time.time()-start))

        ##Augment low resolution data
        #Location of augmentable data

        #Output location for augmented data
        dirs['outDirAI']=dirs['basedir']+dirs['project'] + dirs['tempdirLR'] + '/Augment' + '/regions/'
        dirs['outDirAM']=dirs['basedir']+dirs['project'] + dirs['tempdirLR'] + '/Augment' + '/masks/'

        #Enumerate low resolution class distributions for augmentation ratios
        classDistLR=np.zeros(len(classEnumLR))

        for idx,value in enumerate(classEnumLR):
            classDistLR[idx]=value/sum(classEnumLR)

        #Define number of augmentations per class
        if args.aug_LR > 0:
            augmentOrder=np.argsort(classDistLR)
            classAugs=(np.round(args.aug_LR*(1-classDistLR))+1)
            classAugs=classAugs.astype(int)
            print('Low resolution augmentation distribution:')
            print(classAugs)
            imagesToAugmentLR=dirs['basedir']+dirs['project'] + dirs['tempdirLR'] + 'regions/'
            masksToAugmentLR=dirs['basedir']+dirs['project'] + dirs['tempdirLR'] + 'masks/'
            augmentList=glob(imagesToAugmentLR + '*.jpeg')

            #Parallel iter
            augIter=range(0,len(augmentList))

            auglen=len(augmentList)
            #Augment images in parallel using inverted class distributions for augmentation iterations
            num_cores = multiprocessing.cpu_count()
            start=time.time()

            Parallel(n_jobs=num_cores)(delayed(run_batch)(augmentList,masksToAugmentLR,
                batchidx,classAugs,args.boxSizeLR,args.hbound,args.lbound,
                augmentOrder,dirs,classNum_LR,auglen) for batchidx in augIter)

            moveimages(dirs['outDirAI'], dirs['basedir']+dirs['project'] + '/Permanent/LR/regions/')
            moveimages(dirs['outDirAM'], dirs['basedir']+dirs['project'] + '/Permanent/LR/masks/')

        moveimages(dirs['basedir']+dirs['project'] + dirs['tempdirLR']+ '/regions/', dirs['basedir']+dirs['project'] + '/Permanent/LR/regions/')
        moveimages(dirs['basedir']+dirs['project'] + dirs['tempdirLR']+ '/masks/',dirs['basedir']+dirs['project'] + '/Permanent/LR/masks/')
        end=time.time()-start
        print('Time for low resolution augmenting: ' + str((time.time()-totalStart)/60) + ' minutes.')
        ##High resolution augmentation
        #Enumerate high resolution class distribution

        classDistHR=np.zeros(len(classEnumHR))
        for idx,value in enumerate(classEnumHR):
            classDistHR[idx]=value/sum(classEnumHR)


        #Define number of augmentations per class
        if args.aug_HR >0:
            augmentOrder=np.argsort(classDistHR)
            classAugs=(np.round(args.aug_HR*(1-classDistHR))+1)
            classAugs=classAugs.astype(int)
            print('High resolution augmentation distribution:')
            print(classAugs)
            #High resolution input augmentable data
            imagesToAugmentHR=dirs['basedir']+dirs['project'] + dirs['tempdirHR'] + 'regions/'
            masksToAugmentHR=dirs['basedir']+dirs['project'] + dirs['tempdirHR'] + 'masks/'
            augmentList=glob(imagesToAugmentHR + '*.jpeg')

            #Parallel iterator
            augIter=range(0,len(augmentList))
            auglen=len(augmentList)

            #Output for augmented data
            dirs['outDirAI']=dirs['basedir']+dirs['project'] + dirs['tempdirHR'] + '/Augment' + '/regions/'
            dirs['outDirAM']=dirs['basedir']+dirs['project'] + dirs['tempdirHR'] + '/Augment' + '/masks/'

            #Augment in parallel
            num_cores = multiprocessing.cpu_count()
            start=time.time()
            Parallel(n_jobs=num_cores)(delayed(run_batch)(augmentList,masksToAugmentHR,
                batchidx,classAugs,args.boxSizeHR,args.hbound,args.lbound,
                augmentOrder,dirs,classNum_HR,auglen) for batchidx in augIter)
            end=time.time()-start
            #augamt=len(glob(dirs['outDirAI'] + '*' +  dirs['imExt']))


            moveimages(dirs['outDirAI'], dirs['basedir']+dirs['project'] + '/Permanent/HR/regions/')
            moveimages(dirs['outDirAM'], dirs['basedir']+dirs['project'] + '/Permanent/HR/masks/')

        moveimages(dirs['basedir']+dirs['project'] + dirs['tempdirHR'] + '/regions/', dirs['basedir']+dirs['project'] + '/Permanent/HR/regions/')
        moveimages(dirs['basedir']+dirs['project'] + dirs['tempdirHR'] + '/masks/',dirs['basedir']+dirs['project'] + '/Permanent/HR/masks/')


        #Total time
        print('Time for high resolution augmenting: ' + str((time.time()-totalStart)/60) + ' minutes.')

    #Generate training and validation arguments
    training_args_list = []
    training_args_LR = []
    training_args_HR = []

    ##### LOW REZ ARGS #####
    dirs['outDirAILR']=dirs['basedir']+'/'+dirs['project'] + '/Permanent/LR/regions/'
    dirs['outDirAMLR']=dirs['basedir']+'/'+dirs['project'] + '/Permanent/LR/masks/'

    ########fix this
    trainOutLR=dirs['basedir'] + '/Codes' + '/Deeplab_network/datasetLR/train.txt'
    valOutLR=dirs['basedir'] + '/Codes' + '/Deeplab_network/datasetLR/val.txt'

    generateDatalists(dirs['outDirAILR'],dirs['outDirAMLR'],'/regions/','/masks/',dirs['imExt'],dirs['maskExt'],trainOutLR)
    numImagesLR=len(glob(dirs['outDirAILR'] + '*' + dirs['imExt']))

    numStepsLR=int((args.epoch_LR*numImagesLR)/ args.CNNbatch_sizeLR)
    pretrain_LR=get_pretrain(currentAnnotationIteration,'/LR/',dirs)
    modeldir_LR =dirs['basedir']+dirs['project'] + dirs['modeldir'] + str(currentAnnotationIteration +1) + '/LR/'



    pretrain_HR=get_pretrain(currentAnnotationIteration,'/HR/',dirs)

    modeldir_HR = dirs['basedir']+dirs['project'] + dirs['modeldir'] + str(currentAnnotationIteration+1) + '/HR/'

    # assign to dict
    training_args_LR = {
        'numImages': numImagesLR,
        'data_list': trainOutLR,
        'batch_size': args.CNNbatch_sizeLR,
        'num_steps': numStepsLR,
        'save_interval': np.int(round(numStepsLR/args.saveIntervals)),
        'pretrain_file': pretrain_LR,
        'input_height': args.boxSizeLR,
        'input_width': args.boxSizeLR,
        'modeldir': modeldir_LR,
        'num_classes': classNum_LR,
        'gpu': args.gpu,
        'data_dir': dirs['data_dir_LR'],
        'print_color': "\033[3;37;40m",
        'log_file': modeldir_LR + 'log_'+ str(currentAnnotationIteration+1) +'_LR.txt',
        'log_dir': modeldir_LR + 'log/',
        'learning_rate': args.learning_rate_LR,
        'encoder_name':args.encoder_name
        }
    training_args_list.append(training_args_LR)


    ##### HIGH REZ ARGS #####
    dirs['outDirAIHR']=dirs['basedir']+'/'+dirs['project'] + '/Permanent/HR/regions/'
    dirs['outDirAMHR']=dirs['basedir']+'/'+dirs['project'] + '/Permanent/HR/masks/'

    #######Fix this
    trainOutHR=dirs['basedir'] + '/Codes' +'/Deeplab_network/datasetHR/train.txt'
    valOutHR=dirs['basedir'] + '/Codes' + '/Deeplab_network/datasetHR/val.txt'

    generateDatalists(dirs['outDirAIHR'],dirs['outDirAMHR'],'/regions/','/masks/',dirs['imExt'],dirs['maskExt'],trainOutHR)
    numImagesHR=len(glob(dirs['outDirAIHR'] + '*' + dirs['imExt']))

    numStepsHR=int((args.epoch_HR*numImagesHR)/ args.CNNbatch_sizeHR)
    # assign to dict
    training_args_HR={
        'numImages': numImagesHR,
        'data_list': trainOutHR,
        'batch_size': args.CNNbatch_sizeHR,
        'num_steps': numStepsHR,
        'save_interval': np.int(round(numStepsHR/args.saveIntervals)),
        'pretrain_file': pretrain_HR,
        'input_height': args.boxSizeHR,
        'input_width': args.boxSizeHR,
        'modeldir': modeldir_HR,
        'num_classes': classNum_HR,
        'gpu': args.gpu + args.gpu_num - 1,
        'data_dir': dirs['data_dir_HR'],
        'print_color': "\033[1;32;40m",
        'log_file': modeldir_HR + 'log_'+ str(currentAnnotationIteration+1) +'_HR.txt',
        'log_dir': modeldir_HR + 'log/',
        'learning_rate': args.learning_rate_HR,
        'encoder_name': args.encoder_name
        }
    training_args_list.append(training_args_HR)

    # train networks in parallel
    num_cores = args.gpu_num # GPUs
    Parallel(n_jobs=num_cores, backend='threading')(delayed(train_net)(training_args,dirs) for training_args in training_args_list)



    finish_model_generation(dirs,currentAnnotationIteration)

    print('\n\n\033[92;5mPlease place new wsi file(s) in: \n\t' + dirs['basedir'] + dirs['project']+ dirs['training_data_dir'] + str(currentAnnotationIteration+1))
    print('\nthen run [--option predict]\033[0m\n')




def moveimages(startfolder,endfolder):
    filelist=glob(startfolder + '*')
    for file in filelist:
        fileIDi=file.split('/')[-1]
        move(file,endfolder + fileIDi)

def train_net(training_args,dirs):
    '''
    Recives a dictionary of variables: training_args
    [data_list, num_steps, save_interval, pretrain_file, input_height, input_width, batch_size, num_classes, modeldir, data_dir, gpu]
    '''

    print('Running [' + str( training_args['num_steps'] ), '] iterations')
    print('Saving every [' + str( training_args['save_interval'] ) + '] iterations')

    call(['python3.5', dirs['basedir'] +'/Codes/Deeplab_network/main.py', '--option', 'train',
        '--data_list', training_args['data_list'],
        '--num_steps', str(training_args['num_steps']),
        '--save_interval',str(training_args['save_interval']),
        '--pretrain_file', training_args['pretrain_file'],
        '--input_height',str(training_args['input_height']),
        '--input_width',str(training_args['input_width']),
        '--batch_size',str(training_args['batch_size']),
        '--num_classes',str(training_args['num_classes']),
        '--modeldir', training_args['modeldir'],
        '--data_dir', training_args['data_dir'],
        '--log_file', training_args['log_file'],
        '--log_dir', training_args['log_dir'],
        '--gpu', str(training_args['gpu']),
        '--learning_rate', str(training_args['learning_rate']),
        '--print_color', training_args['print_color'],
        '--encoder_name',training_args['encoder_name']])


def check_model_generation(dirs):
    modelsCurrent=os.listdir(dirs['basedir'] + dirs['project'] + dirs['modeldir'])
    gens=map(int,modelsCurrent)
    modelOrder=np.sort(list(gens))[::-1]

    for idx in modelOrder:
        modelsChkptsLR=glob(dirs['basedir'] + dirs['project'] + dirs['modeldir']+str(idx) + '/LR/*.ckpt*')
        modelsChkptsHR=glob(dirs['basedir'] + dirs['project'] + dirs['modeldir']+ str(idx) +'/HR/*.ckpt*')

        if modelsChkptsLR == []:
            continue
        elif modelsChkptsHR == []:
            continue
        else:
            return idx
            break

def finish_model_generation(dirs,currentAnnotationIteration):
    make_folder(dirs['basedir'] + dirs['project'] + dirs['training_data_dir'] + str(currentAnnotationIteration + 1))

def get_pretrain(currentAnnotationIteration,res,dirs):

    if currentAnnotationIteration==0:
        pretrain_file = glob(dirs['basedir']+dirs['project'] + dirs['modeldir'] + str(currentAnnotationIteration) + res + '*')
        pretrain_file=pretrain_file[0].split('.')[0] + '.' + pretrain_file[0].split('.')[1]

    else:
        pretrains=glob(dirs['basedir']+dirs['project'] + dirs['modeldir'] + str(currentAnnotationIteration) + res + 'model*')
        print(pretrains)
        maxmodel=0
        for modelfiles in pretrains:
            modelID=modelfiles.split('.')[-2].split('-')[1]
            if int(modelID)>maxmodel:
                maxmodel=int(modelID)
        pretrain_file=dirs['basedir']+dirs['project'] + dirs['modeldir'] + str(currentAnnotationIteration) + res + 'model.ckpt-' + str(maxmodel)
    return pretrain_file

def restart_line(): # for printing chopped image labels in command line
    sys.stdout.write('\r')
    sys.stdout.flush()

def file_len(fname): # get txt file length (number of lines)
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def make_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory) # make directory if it does not exit already # make new directory # Check if folder exists, if not make it

def make_all_folders(dirs):


    make_folder(dirs['basedir'] +dirs['project']+ dirs['tempdirLR'] + '/regions')
    make_folder(dirs['basedir'] +dirs['project']+ dirs['tempdirLR'] + '/masks')

    make_folder(dirs['basedir'] +dirs['project']+ dirs['tempdirLR'] + '/Augment' +'/regions')
    make_folder(dirs['basedir'] +dirs['project']+ dirs['tempdirLR'] + '/Augment' +'/masks')

    make_folder(dirs['basedir']+dirs['project'] + dirs['tempdirHR'] + '/regions')
    make_folder(dirs['basedir'] +dirs['project']+ dirs['tempdirHR'] + '/masks')

    make_folder(dirs['basedir']+dirs['project'] + dirs['tempdirHR'] + '/Augment' +'/regions')
    make_folder(dirs['basedir']+dirs['project']+ dirs['tempdirHR'] + '/Augment' +'/masks')

    make_folder(dirs['basedir'] +dirs['project']+ dirs['modeldir'])
    make_folder(dirs['basedir'] +dirs['project']+ dirs['training_data_dir'])


    make_folder(dirs['basedir'] +dirs['project']+ '/Permanent' +'/LR/'+ 'regions/')
    make_folder(dirs['basedir'] +dirs['project']+ '/Permanent' +'/LR/'+ 'masks/')
    make_folder(dirs['basedir'] +dirs['project']+ '/Permanent' +'/HR/'+ 'regions/')
    make_folder(dirs['basedir'] +dirs['project']+ '/Permanent' +'/HR/'+ 'masks/')

    make_folder(dirs['basedir'] +dirs['project']+ dirs['training_data_dir'])

    make_folder(dirs['basedir'] + '/Codes/Deeplab_network/datasetLR')
    make_folder(dirs['basedir'] + '/Codes/Deeplab_network/datasetHR')

def return_region(args, wsi_mask, wsiID, fileID, yStart, xStart, idxy, idxx, downsampleRate, outdirT, region_size, dirs, chop_regions,classNum): # perform cutting in parallel
    sys.stdout.write('   <'+str(xStart)+'/'+ str(yStart)+ '>   ')
    sys.stdout.flush()
    restart_line()

    if chop_regions[idxy,idxx] != 0:

        uniqID=fileID + str(yStart) + str(xStart)
        if wsiID.split('.')[-1] != 'tif':
            slide=getWsi(wsiID)
            Im=np.array(slide.read_region((xStart,yStart),0,(region_size,region_size)))
            Im=Im[:,:,:3]
        else:
            yEnd = yStart + region_size
            xEnd = xStart + region_size
            Im = np.zeros([region_size,region_size,3], dtype=np.uint8)
            Im_ = imread(wsiID)[yStart:yEnd, xStart:xEnd, :3]
            Im[0:Im_.shape[0], 0:Im_.shape[1], :] = Im_

        mask_annotation=wsi_mask[yStart:yStart+region_size,xStart:xStart+region_size]

        o1,o2=mask_annotation.shape
        if o1 !=region_size:
            mask_annotation=np.pad(mask_annotation,((0,region_size-o1),(0,0)),mode='constant')
        if o2 !=region_size:
            mask_annotation=np.pad(mask_annotation,((0,0),(0,region_size-o2)),mode='constant')

        if downsampleRate !=1:
            c=(Im.shape)
            s1=int(c[0]/(downsampleRate**.5))
            s2=int(c[1]/(downsampleRate**.5))
            Im=(resize(Im,(s1,s2),mode='reflect')*255).astype('uint8')
            mask_annotation=resize(mask_annotation,(s1,s2),order=0,preserve_range=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imsave(outdirT + '/regions/' + uniqID + dirs['imExt'],Im)
            imsave(outdirT + '/masks/' + uniqID +dirs['maskExt'],np.uint8(mask_annotation))
        classespresent=np.unique(mask_annotation)
        classes=range(0,classNum)
        classEnumC=np.zeros([classNum,1])

        for index,chk in enumerate(classes):
            if chk in classespresent:
                classEnumC[index]=classEnumC[index]+1
        return classEnumC
    else:


        classes=range(0,classNum)
        classEnumC=np.zeros([classNum,1])
        return classEnumC

def load_batch(imageList,maskDir,batchindex,batch_augs,boxsize,dirs):

    X_data=[]
    mask_data=[]
    for b in range(0,batch_augs):
        fileIDi = imageList[batchindex]
        X_data.append(imread(fileIDi))
        fileIDi=fileIDi.split('/')[-1].split('.')[0]
        mask_data.append(imread(maskDir+fileIDi+dirs['maskExt']))


    return X_data,mask_data #Load N copies of current image based on class distributions

def save_batch(imageblock,maskblock,imageList,batchindex,dirs):

    fileIDi = imageList[batchindex]
    fileIDi=fileIDi.split('/')[-1].split('.')[0]
    for index in range(0,len(imageblock)):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imsave(dirs['outDirAI'] + fileIDi +'_'+ str(index) + dirs['imExt'],np.uint8(imageblock[index]*255))
            imsave(dirs['outDirAM'] + fileIDi +'_'+ str(index) + dirs['maskExt'],np.uint8(maskblock[index])) #Save N copies of current image

def run_batch(imageList, maskDir, batchindex, class_augs, box_size,
    hbound, lbound, augmentOrder,dirs,classNum,auglen):
    sys.stdout.write('   <'+str(batchindex)+'/'+str(auglen)+ '>   ')
    sys.stdout.flush()
    restart_line()
    #Load image, determine augmentation probability, augment image, augment colorspace, save images
    global seq
    seq_det = seq.to_deterministic()

    imageblock,maskblock=load_batch(imageList,maskDir,batchindex,1,box_size,dirs)

    classespresent=np.unique(maskblock)
    classes=range(0,classNum)

    for idx in augmentOrder:
        if idx in classespresent:
            prob=class_augs[idx]
            break
    imageblock,maskblock=load_batch(imageList,maskDir,batchindex,prob,box_size,dirs)


    imageblock=seq_det.augment_images(imageblock)
    imageblock=colorshift(imageblock,hbound,lbound)

    maskblock=seq_det.augment_images(maskblock)
    save_batch(imageblock,maskblock,imageList,batchindex,dirs)

def colorshift(imageblock, hbound, lbound): #Shift Hue of HSV space and Lightness of LAB space
    for im in range(0,len(imageblock)):
        hShift=np.random.normal(0,hbound)
        lShift=np.random.normal(1,lbound)
        imageblock[im]=randomHSVshift(imageblock[im],hShift,lShift)
    return imageblock
