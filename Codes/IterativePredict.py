import cv2
import numpy as np
import os
import sys
import argparse
import multiprocessing
import lxml.etree as ET
import warnings
import time

sys.path.append(os.getcwd()+'/Codes')

from glob import glob
from subprocess import call
from joblib import Parallel, delayed
from skimage.io import imread, imsave
from skimage.transform import resize
from scipy.ndimage.measurements import label
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects
from skimage import color
from shutil import rmtree
from IterativeTraining import get_num_classes
from get_choppable_regions import get_choppable_regions
from get_network_performance import get_perf
from matplotlib import pyplot as plt


# define xml class colormap
xml_color = [65280, 65535, 255, 16711680, 33023]

def validate(args):
    # define folder structure dict
    dirs = {'outDir': args.base_dir + '/' + args.project + args.outDir}
    dirs['txt_save_dir'] = '/txt_files/'
    dirs['img_save_dir'] = '/img_files/'
    dirs['final_output_dir'] = '/boundaries/'
    dirs['final_boundary_image_dir'] = '/images/'
    dirs['mask_dir'] = '/wsi_mask/'
    dirs['chopped_dir'] = '/originals/'
    dirs['crop_dir'] = '/wsi_crops/'
    dirs['save_outputs'] = args.save_outputs
    dirs['modeldir'] = '/MODELS/'
    dirs['training_data_dir'] = '/TRAINING_data/'
    dirs['validation_data_dir'] = '/HOLDOUT_data/'

    # find current iteration
    if args.iteration == 'none':
        iteration = get_iteration(args=args)
    else:
        iteration = int(args.iteration)

    # get all WSIs
    WSIs = []
    for ext in [args.wsi_ext]:
        WSIs.extend(glob(args.base_dir + '/' + args.project + dirs['validation_data_dir'] + '/*' + ext))

    if iteration == 'none':
        print('ERROR: no trained models found \n\tplease use [--option train]')

    else:
        for iter in range(1,iteration+1):
            dirs['xml_save_dir'] = args.base_dir + '/' + args.project + dirs['validation_data_dir'] + str(iter) + '_Predicted_XMLs/'


            # check main directory exists
            make_folder(dirs['outDir'])

            if not os.path.exists(dirs['xml_save_dir']):
                make_folder(dirs['xml_save_dir'])
                print('working on iteration: ' + str(iter))

                with open(args.base_dir + '/' + args.project + dirs['validation_data_dir'] + 'validation_stats.txt', 'a') as f:
                    f.write('\niteration: \t'+str(iter)+'\n')
                    f.write('\twsi\t\t\tsensitivity\t\t\tspecificity\t\t\tprecision\t\t\taccuracy\t\t\tprediction time\n')

                for wsi in WSIs:
                    # predict xmls
                    startTime = time.time()
                    predict_xml(args=args, dirs=dirs, wsi=wsi, iteration=iter)
                    predictTime = time.time() - startTime
                    # test performance
                    gt_xml = os.path.splitext(wsi)[0] + '.xml'
                    predicted_xml = gt_xml.split('/')
                    predicted_xml = dirs['xml_save_dir'] + predicted_xml[-1]
                    sensitivity,specificity,precision,accuracy = get_perf(wsi=wsi, xml1=gt_xml, xml2 = predicted_xml, args=args)

                    with open(args.base_dir + '/' + args.project + dirs['validation_data_dir'] + 'validation_stats.txt', 'a') as f:
                        f.write('\t'+wsi.split('/')[-1]+'\t\t'+str(sensitivity)+'\t\t'+str(specificity)+'\t\t'+str(precision)+'\t\t'+str(accuracy)+'\t\t'+str(predictTime)+'\n')

        print('\n\n\033[92;5mDone validating: \n\t\033[0m\n')

def predict(args):
    # define folder structure dict
    dirs = {'outDir': args.base_dir + '/' + args.project + args.outDir}
    dirs['txt_save_dir'] = '/txt_files/'
    dirs['img_save_dir'] = '/img_files/'
    dirs['final_output_dir'] = '/boundaries/'
    dirs['final_boundary_image_dir'] = '/images/'
    dirs['mask_dir'] = '/wsi_mask/'
    dirs['chopped_dir'] = '/originals/'
    dirs['crop_dir'] = '/wsi_crops/'
    dirs['save_outputs'] = args.save_outputs
    dirs['modeldir'] = '/MODELS/'
    dirs['training_data_dir'] = '/TRAINING_data/'

    # find current iteration
    if args.iteration == 'none':
        iteration = get_iteration(args=args)
    else:
        iteration = int(args.iteration)

    print(iteration)
    dirs['xml_save_dir'] = args.base_dir + '/' + args.project + dirs['training_data_dir'] + str(iteration) + '/Predicted_XMLs/'

    if iteration == 'none':
        print('ERROR: no trained models found \n\tplease use [--option train]')

    else:
        # check main directory exists
        make_folder(dirs['outDir'])
        make_folder(dirs['xml_save_dir'])

        # get all WSIs
        WSIs = []
        for ext in [args.wsi_ext]:

            WSIs.extend(glob(args.base_dir + '/' + args.project + dirs['training_data_dir'] + str(iteration)+ '/*' + ext))

        for wsi in WSIs:
            #try:
                predict_xml(args=args, dirs=dirs, wsi=wsi, iteration=iteration)
            #except KeyboardInterrupt:
            #    break
            #except:
            #    print('!!! Prediction on ' + wsi + ' failed\nmoving on...')
        print('\n\n\033[92;5mPlease correct the xml annotations found in: \n\t' + dirs['xml_save_dir'])
        print('\nthen place them in: \n\t'+ args.base_dir + '/' + args.project + dirs['training_data_dir'] + str(iteration) + '/')
        print('\nand run [--option train]\033[0m\n')


def predict_xml(args, dirs, wsi, iteration):
    # reshape regions calc
    downsample = int(args.downsampleRateLR**.5)
    downsample_HR = int(args.downsampleRateHR**.5)
    region_size = int(args.boxSizeLR*(downsample))
    step = int(region_size*(1-args.overlap_percentLR))


    # figure out the number of classes
    if args.classNum == 0:
        annotatedXMLs=glob(args.base_dir + '/' + args.project + dirs['training_data_dir'] + str(iteration-1) + '/*.xml')
        classes = []
        for xml in annotatedXMLs:
            classes.append(get_num_classes(xml))
        print(classes)
        print(args.classNum)
        classNum_LR = max(classes)
        classNum_HR = max(classes)
    else:
        classNum_LR = args.classNum
        if args.classNum_HR != 0:
            classNum_HR = args.classNum_HR
        else:
            classNum_HR = classNum_LR

    # chop wsi
    if args.chop_data == 'True':
        # chop wsi
        fileID, test_num_steps = chop_suey(wsi, dirs, downsample, region_size, step, args)
        dirs['fileID'] = fileID
        print('Chop SUEY!\n')
    else:
        basename = os.path.splitext(wsi)[0]

        if wsi.split('.')[-1] != 'tif':
            slide=getWsi(wsi)
            # get image dimensions
            dim_x, dim_y=slide.dimensions
        else:
            im = Image.open(wsi)
            dim_x, dim_y=im.size

        fileID=basename.split('/')
        dirs['fileID'] = fileID=fileID[len(fileID)-1]
        test_num_steps = file_len(dirs['outDir'] + fileID + dirs['txt_save_dir'] + fileID + '_images' + ".txt")

    # call DeepLab for prediction (Low resolution)
    print('finding Glom locations ...\n')

    make_folder(dirs['outDir'] + fileID + dirs['img_save_dir'] + 'prediction')

    test_data_list = fileID + '_images' + '.txt'
    modeldir = args.base_dir + '/' + args.project + dirs['modeldir'] + str(iteration) + '/LR'
    test_step = get_test_step(modeldir)
    print("\033[1;32;40m"+"starting prediction using model: \n\t" + modeldir + str(test_step) + "\033[0;37;40m"+"\n\n")

    call(['python3.5', args.base_dir+'/Codes/Deeplab_network/main.py',
        '--option', 'predict',
        '--test_data_list', dirs['outDir']+fileID+dirs['txt_save_dir']+test_data_list,
        '--out_dir', dirs['outDir']+fileID+dirs['img_save_dir'],
        '--test_step', str(test_step),
        '--test_num_steps', str(test_num_steps),
        '--modeldir', modeldir,
        '--data_dir', dirs['outDir']+fileID+dirs['img_save_dir'],
        '--num_classes', str(classNum_LR),
        '--gpu', str(args.gpu),
        '--encoder_name',args.encoder_name])

    # un chop
    print('\nreconstructing wsi map ...\n')
    wsiMask = un_suey(dirs=dirs, args=args)


    # save hotspots
    if dirs['save_outputs'] == True:
    	#reduce the resolution of the image
        wsidims=wsiMask.shape
        wsiMask_save=resize(wsiMask,(int(wsidims[0]/4),int(wsidims[1]/4)),order=0,preserve_range=True)

        make_folder(dirs['outDir'] + fileID + dirs['mask_dir'])
        print('saving to: ' + dirs['outDir'] + fileID + dirs['mask_dir'] + fileID  + '.png')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imsave(dirs['outDir'] + fileID + dirs['mask_dir'] + fileID + '.png', wsiMask_save*255)

    # find glom locations in reconstructed map
    print('\ninterpreting prediction map ...')
    test_num_steps, labeledArray, label_offsets = find_suey(wsiMask, dirs, downsample, args, wsi)
    print('\n\nthe cropped regions have been saved to: ' + dirs['outDir'] + fileID + dirs['img_save_dir'] + fileID + dirs['crop_dir'])

    # call DeepLab to predict Glom boundaries (High resolution)
    print('\ngetting Glom boundaries ...\n')
    make_folder(dirs['outDir'] + fileID + dirs['final_output_dir'] + 'prediction')

    test_data_list = fileID + '_crops.txt'
    modeldir = args.base_dir + '/' + args.project + dirs['modeldir'] + str(iteration) + '/HR'
    test_step = get_test_step(modeldir)
    print("\033[1;32;40m"+"starting prediction using model: \n\t" + modeldir + str(test_step) + "\033[0;37;40m"+"\n\n")

    call(['python3.5', args.base_dir+'/Codes/Deeplab_network/main.py',
        '--option', 'predict',
        '--test_data_list', dirs['outDir']+fileID+dirs['txt_save_dir']+test_data_list,
        '--out_dir', dirs['outDir']+fileID+dirs['final_output_dir'],
        '--test_step', str(test_step),
        '--test_num_steps', str(test_num_steps),
        '--modeldir', modeldir,
        '--data_dir', dirs['outDir']+fileID+dirs['img_save_dir'],
        '--num_classes', str(classNum_HR),
        '--gpu', str(args.gpu),
        '--encoder_name',args.encoder_name])

    print('\nsaving final images ...')
    print('\nstitching high resolution WSI mask:')

    wsiMask_HR=unstitch_HR(dirs=dirs,args=args,wsi=wsi)
    wsidims=wsiMask_HR.shape
    d1=int(wsidims[0]*(1/args.approx_downsample))
    d2=int(wsidims[1]*(1/args.approx_downsample))

    if args.approx_downsample!=1:
        print('\nDownsampling high resolution mask for prediction smoothing...')
        wsiMask_HR=resize(wsiMask_HR,(d1,d2),order=0,preserve_range=True)
    #mask_display=resize(wsiMask_HR,(d1,d2))
    #plt.imshow(mask_display/np.max(wsiMask))
    #plt.show()
    print('\nGenerating XML annotations from WSI mask...')
    make_folder(dirs['outDir'] + dirs['fileID'] + dirs['final_output_dir'] + dirs['final_boundary_image_dir'])
    crop_suey(wsiMask_HR,label_offsets, dirs, args, classNum_HR, downsample_HR)


    # clean up
    if dirs['save_outputs'] == False:
        print('\ncleaning up')
        rmtree(dirs['outDir']+fileID)


def get_iteration(args):
    currentmodels=os.listdir(args.base_dir + '/' + args.project + '/MODELS/')
    if not currentmodels:
        return 'none'
    else:
        currentmodels=map(int,currentmodels)
        Iteration=np.max(list(currentmodels))
        return Iteration

def get_test_step(modeldir):
    pretrains=glob(modeldir + '/*.ckpt*')
    maxmodel=0
    for modelfiles in pretrains:
        modelID=modelfiles.split('.')[-2].split('-')[1]
        try:
            modelID = int(modelID)
            if modelID>maxmodel:
                maxmodel=modelID
        except: pass
    return maxmodel

def make_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory) # make directory if it does not exit already # make new directory

def restart_line(): # for printing chopped image labels in command line
    sys.stdout.write('\r')
    sys.stdout.flush()

def getWsi(path): #imports a WSI
    import openslide
    slide = openslide.OpenSlide(path)
    return slide

def file_len(fname): # get txt file length (number of lines)
    with open(fname) as f:
        for i, l in enumerate(f):
            pass

    if 'i' in locals():
        return i + 1

    else:
        return 0


def chop_suey(wsi, dirs, downsample, region_size, step, args): # chop wsi
    print('\nopening: ' + wsi)
    basename = os.path.splitext(wsi)[0]

    slide=getWsi(wsi)

    fileID=basename.split('/')
    dirs['fileID'] = fileID=fileID[len(fileID)-1]
    print('\nchopping ...\n')

    # make txt file
    make_folder(dirs['outDir'] + fileID + dirs['txt_save_dir'])
    f_name = dirs['outDir'] + fileID + dirs['txt_save_dir'] + fileID + ".txt"
    f2_name = dirs['outDir'] + fileID + dirs['txt_save_dir'] + fileID + '_images' + ".txt"
    f = open(f_name, 'w')
    f2 = open(f2_name, 'w')
    f2.close()

    make_folder(dirs['outDir'] + fileID + dirs['img_save_dir'] + dirs['chopped_dir'])

    # get image dimensions
    dim_x, dim_y=slide.dimensions
    f.write('Image dimensions:\n')

    # make index for iters
    index_y=range(0,dim_y-step,step)
    index_x=range(0,dim_x-step,step)

    f.write('X dim: ' + str((index_x[-1]+region_size)/downsample) +'\n')
    f.write('Y dim: ' + str((index_y[-1]+region_size)/downsample) +'\n\n')
    f.write('Regions:\n')
    f.write('image:xStart:xStop:yStart:yStop\n\n')
    f.close()

    # get non white regions
    choppable_regions = get_choppable_regions(wsi=wsi, index_x=index_x, index_y=index_y, boxSize=region_size,white_percent=args.white_percent)

    print('saving region:')

    num_cores = multiprocessing.cpu_count()

    Parallel(n_jobs=num_cores, backend='threading')(delayed(chop_wsi)(yStart=i, xStart=j, idxx=idxx, idxy=idxy, f_name=f_name, f2_name=f2_name, dirs=dirs, downsample=downsample, region_size=region_size, args=args, wsi=wsi, choppable_regions=choppable_regions) for idxy, i in enumerate(index_y) for idxx, j in enumerate(index_x))

    test_num_steps = file_len(dirs['outDir'] + fileID + dirs['txt_save_dir'] + fileID + '_images' + ".txt")
    print('\n\n' + str(test_num_steps) +' image regions chopped')

    return fileID, test_num_steps

def chop_wsi(yStart, xStart, idxx, idxy, f_name, f2_name, dirs, downsample, region_size, args, wsi, choppable_regions): # perform cutting in parallel
    if choppable_regions[idxy, idxx] != 0:
        slide = getWsi(wsi)

        yEnd = yStart+region_size
    	#print(yEnd)
        xEnd = xStart+region_size
    	#print(xEnd)
        xLen=xEnd-xStart
        yLen=yEnd-yStart

        subsect= np.array(slide.read_region((xStart,yStart),0,(xLen,yLen)))
        subsect=subsect[:,:,:3]

		#print(whiteRatio)
        imageIter = str(xStart)+str(yStart)

        f = open(f_name, 'a+')
        f2 = open(f2_name, 'a+')

        # append txt file
        f.write(imageIter + ':' + str(xStart/downsample) + ':' + str(xEnd/downsample)
            + ':' + str(yStart/downsample) + ':' + str(yEnd/downsample) + '\n')

		# resize images ans masks
        c=(subsect.shape)
        s1=int(c[0]/(args.downsampleRateLR**.5))
        s2=int(c[1]/(args.downsampleRateLR**.5))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            subsect=(resize(subsect,(s1,s2), mode='constant')*255).astype('uint8')

        # save image
        directory = dirs['outDir'] + dirs['fileID'] + dirs['img_save_dir'] + dirs['chopped_dir']
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imsave(directory + dirs['fileID'] + str(imageIter) + args.imBoxExt,subsect)

        f2.write(dirs['chopped_dir'] + dirs['fileID'] + str(imageIter) + args.imBoxExt + '\n')
        f.close()
        f2.close()

        sys.stdout.write('   <'+str(xStart)+':'+str(xEnd)+' '+str(yStart)+':'+str(yEnd)+'>   ')
        sys.stdout.flush()
        restart_line()

def unstitch_HR(dirs,args,wsi):# reconstruct wsi from predicted masks, high resolution
    image_coordinate_file=dirs['outDir'] + dirs['fileID'] + dirs['txt_save_dir'] + dirs['fileID'] + '_crops.txt'
    slide=getWsi(wsi)
    dim_x, dim_y=slide.dimensions
    wsiMask=np.zeros([dim_y,dim_x])
    f = open(image_coordinate_file, 'r')
    lines = f.readlines()
    f.close()
    lines = np.array(lines)

    for regionNum in range(0, np.size(lines)):

        sys.stdout.write('   <'+str(regionNum)+'/'+ str(np.size(lines)-1)+ '>   ')
        sys.stdout.flush()
        restart_line()
        # get region
        uniqueImageID=lines[regionNum].split('/')[-1]
        maskID=uniqueImageID.split('.')[0]
        region = maskID.split('_')[-4:]
        # read mask
        mask = imread(dirs['outDir'] + dirs['fileID'] + dirs['final_output_dir'] + 'prediction/'+ maskID + '_mask.png')

        # get region bounds
        xStart = np.int32(region[2])
        #print('xStart: ' + str(xStart))
        xStop = np.int32(region[3])
        #print('xStop: ' + str(xStop))
        yStart = np.int32(region[0])
        if yStart < 0:
            yStart = 0
        #print('yStart: ' + str(yStart))
        yStop = np.int32(region[1])
        #print('yStop: ' + str(yStop))

        mask_part = wsiMask[yStart:yStop, xStart:xStop]
        ylen, xlen = np.shape(mask_part)
        mask = mask[:ylen, :xlen]

        # populate wsiMask with max
        #print(np.shape(wsiMask))
        wsiMask[yStart:yStop, xStart:xStop] = np.maximum(mask_part, mask)
        #wsiMask[yStart:yStop, xStart:xStop] = np.ones([yStop-yStart, xStop-xStart])

    return wsiMask

def un_suey(dirs, args): # reconstruct wsi from predicted masks, low resolution
    txtFile = dirs['fileID'] + '.txt'

    # read txt file
    f = open(dirs['outDir'] + dirs['fileID'] + dirs['txt_save_dir'] + txtFile, 'r')
    lines = f.readlines()
    f.close()
    lines = np.array(lines)

    # get wsi size
    xDim = np.int32(float((lines[1].split(': ')[1]).split('\n')[0]))
    yDim = np.int32(float((lines[2].split(': ')[1]).split('\n')[0]))
    #print('xDim: ' + str(xDim))
    #print('yDim: ' + str(yDim))

    # make wsi mask
    wsiMask = np.zeros([yDim, xDim])

    # read image regions
    for regionNum in range(7, np.size(lines)):
        # get region
        region = lines[regionNum].split(':')
        region[4] = region[4].split('\n')[0]

        # read mask
        mask = imread(dirs['outDir'] + dirs['fileID'] + dirs['img_save_dir'] + 'prediction/' + dirs['fileID'] + region[0] + '_mask.png')

        # get region bounds
        xStart = np.int32(float(region[1]))
        #print('xStart: ' + str(xStart))
        xStop = np.int32(float(region[2]))
        #print('xStop: ' + str(xStop))
        yStart = np.int32(float(region[3]))
        if yStart < 0:
            yStart = 0
        #print('yStart: ' + str(yStart))
        yStop = np.int32(float(region[4]))
        #print('yStop: ' + str(yStop))

        # populate wsiMask with max
        #print(np.shape(wsiMask))
        wsiMask[yStart:yStop, xStart:xStop] = np.maximum(wsiMask[yStart:yStop, xStart:xStop], mask)
        #wsiMask[yStart:yStop, xStart:xStop] = np.ones([yStop-yStart, xStop-xStart])

    return wsiMask

def find_suey(wsiMask, dirs, downsample, args, wsi): # locates the detected glom regions in the reconstructed wsi mask
    # clean up mask
    print('   removing small objects')
    cleanMask = remove_small_objects(wsiMask.astype(bool), (args.min_size)/(downsample*downsample))

    # find all unconnected regions
    labeledArray, num_features = label(cleanMask)
    print('found: '+ str(num_features-1) + '  regions')

    # save cleaned mask
    if dirs['save_outputs'] == True:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imsave(dirs['outDir'] + dirs['fileID'] + dirs['mask_dir'] + dirs['fileID'] + '_cleaned.png', cleanMask*255)

    make_folder(dirs['outDir'] + dirs['fileID'] + dirs['img_save_dir'] + dirs['crop_dir'])
    f_name = dirs['outDir'] + dirs['fileID'] + dirs['txt_save_dir'] + dirs['fileID'] + '_crops.txt'
    f = open(f_name, 'w')
    f.close()

    # run crop_region in parallel
    print('\nsaving:')
    #num_cores = multiprocessing.cpu_count()
    #Parallel(n_jobs=num_cores)(delayed(crop_region)(region_iter=i, labeledArray=labeledArray, fileID=fileID, f_name=f_name) for i in range(1, num_features))
    label_offsets = []
    for region_iter in range(1, num_features+1):
        label_offsets = crop_region(region_iter=region_iter, labeledArray=labeledArray, f_name=f_name, dirs=dirs, downsample=downsample, args=args, wsi=wsi,label_offsets=label_offsets)
        #label_offsets.append(label_offset[:])

    test_num_steps = file_len(dirs['outDir'] + dirs['fileID'] + dirs['txt_save_dir'] + dirs['fileID'] + '_crops' + ".txt")
    return test_num_steps, labeledArray, label_offsets

def crop_region(region_iter, labeledArray, f_name, dirs, downsample, args, wsi,label_offsets): # crop selected region from wsi and save // location defined by labeledArray
    slide = getWsi(wsi)
    slide_size = slide.dimensions

    # get list of locations for pixels == region_iter
    mask_region = np.argwhere(labeledArray == region_iter)
    # calculate the region bounds
    yStart = max(0, (min(mask_region[:,0]) * downsample) - args.LR_region_pad)
    yLen = min((max(mask_region[:,0]) * downsample) - yStart + args.LR_region_pad, slide_size[1]*downsample - yStart)
    xStart = max(0, (min(mask_region[:,1]) * downsample) - args.LR_region_pad)
    xLen = min((max(mask_region[:,1]) * downsample) - xStart + args.LR_region_pad, slide_size[0]*downsample - yStart)

    region = np.array(slide.read_region((xStart,yStart),0,(xLen,yLen)))
    region = region[:,:,0:3]

    dims=region.shape
    max_block_dim=args.max_block_dim
    sub_box_size=args.boxSizeHR
    sub_region_iter=0
    if (xLen*yLen)>(max_block_dim*max_block_dim):
        stepHR = int(sub_box_size*(1-args.overlap_percentHR))
        Idx1=np.array(range(xStart,xStart+xLen+sub_box_size,stepHR)-xStart)
        Idx2=np.array(range(yStart,yStart+yLen+sub_box_size,stepHR)-yStart)
        Idx1_o=np.array(range(xStart,xStart+xLen+sub_box_size,stepHR))
        Idx2_o=np.array(range(yStart,yStart+yLen+sub_box_size,stepHR))

        Ovl1=xLen-Idx1[-2]
        Ovl2=yLen-Idx2[-2]
        Idx1[-1]=xLen
        Idx2[-1]=yLen
        Idx1_o[-1]=xStart+xLen
        Idx2_o[-1]=yStart+yLen

        for index1,ii in enumerate(Idx1):
            for index2,jj in enumerate(Idx2):
                sys.stdout.write('   <' + str(region_iter)+'_' +str(sub_region_iter) + '>   ')
                sys.stdout.flush()
                restart_line()
                if ii==Idx1[-1]:
                    continue
                if jj==Idx2[-1]:
                    continue
                if ii>xLen:
                    continue
                if jj>yLen:
                    continue

                if (ii+sub_box_size)>xLen:
                    IdxEndx=xLen
                else:
                    IdxEndx=ii+sub_box_size

                if (jj+sub_box_size)>yLen:
                    IdxEndy=yLen
                else:
                    IdxEndy=jj+sub_box_size

                im_name=dirs['fileID']+'_'+str(Idx2_o[index2])+'_'+str(IdxEndy+yStart)+'_'+str(Idx1_o[index1])+'_'+str(IdxEndx+xStart)+args.imBoxExt
                sub_block=region[jj:IdxEndy,ii:IdxEndx,:]
                sub_region_iter=sub_region_iter+1

                f = open(f_name, 'a+')
                f.write(dirs['crop_dir'] + im_name+ '\n')
                f.close()

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    imsave(dirs['outDir'] + dirs['fileID'] + dirs['img_save_dir'] + dirs['crop_dir'] + im_name, sub_block)

                label_offsets.append({'Y':Idx2_o[index2],'X': Idx1_o[index1]})
    else:
        sub_region_iter=0
        # print output
        sys.stdout.write('   <' + str(region_iter)+'_' +str(sub_region_iter) + '>   ')
        sys.stdout.flush()
        restart_line()
        im_name=dirs['fileID']+'_'+str(yStart)+'_'+str(yStart+yLen)+'_'+str(xStart)+'_'+str(xStart+xLen)+args.imBoxExt
        # write image path to text file
        f = open(f_name, 'a+')
        f.write(dirs['crop_dir'] + im_name + '\n')
        f.close()


        # save image region
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imsave(dirs['outDir'] + dirs['fileID'] + dirs['img_save_dir'] + dirs['crop_dir'] + im_name, region)
        label_offsets.append({'Y': yStart, 'X': xStart})
    return label_offsets


def crop_suey(wsiMask,label_offsets, dirs, args, classNum, downsample):
    # make xml
    Annotations = xml_create()
    # add annotation
    for i in range(classNum)[1:]: # exclude background class
        Annotations = xml_add_annotation(Annotations=Annotations, annotationID=i)


    for classregion in range(1,classNum):
        binaryMask = np.zeros(np.shape(wsiMask)).astype('uint8')
        binaryMask[wsiMask == classregion] = 1
        pointsList = get_contour_points(binaryMask, args=args, downsample=downsample)

        for i in range(np.shape(pointsList)[0]):
            pointList = pointsList[i]

            Annotations = xml_add_region(Annotations=Annotations, pointList=pointList,args=args, annotationID=classregion)

    #txtFile = dirs['fileID'] + '_crops.txt'


    #make_folder(dirs['outDir'] + dirs['fileID'] + dirs['final_output_dir'] + dirs['final_boundary_image_dir'][1:])
    txtFile = dirs['fileID'] + '_crops.txt'

    # read txt file with img paths
    lines=[]
    f = open(dirs['outDir'] + dirs['fileID'] + dirs['txt_save_dir'] + txtFile, 'r')
    lines = f.readlines()
    f.close()
    lines = np.array(lines)

    for line in range(0, np.size(lines)):
        image_path = lines[line].split('\n')[0]

        file_name = (image_path.split('.')[0]).split(dirs['crop_dir'])[1]
        mask_image = imread(dirs['outDir'] + dirs['fileID'] + dirs['final_output_dir'] + 'prediction/'
            + file_name + '_mask.png')
        # save mask images
        if dirs['save_outputs'] == True:
            glom_image = imread(dirs['outDir'] + dirs['fileID'] + dirs['img_save_dir'] + image_path[1:])
            if np.sum(mask_image) != 0:
                # remove background in images
                for i in range(3):
                    glom_image[:,:,i] = glom_image[:,:,i] * (mask_image * ((1-args.bg_intensity)) + args.bg_intensity)

                    # save resulting image
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        imsave(dirs['outDir'] + dirs['fileID'] + dirs['final_output_dir'] + dirs['final_boundary_image_dir'][1:] + file_name + args.finalImgExt, glom_image)

    # save xml
    xml_save(Annotations=Annotations, filename=dirs['xml_save_dir']+'/'+dirs['fileID']+'.xml')

def get_contour_points(mask, args, downsample, offset={'X': 0,'Y': 0}):
    # returns a dict pointList with point 'X' and 'Y' values
    # input greyscale binary image
    #_, maskPoints, contours = cv2.findContours(np.array(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    _,maskPoints, contours = cv2.findContours(np.array(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

    pointsList = []
    for j in range(np.shape(maskPoints)[0]):
        if cv2.contourArea(maskPoints[j]) > ((args.min_size)/(downsample*downsample*args.approx_downsample)):
            pointList = []
            for i in range(0,np.shape(maskPoints[j])[0]):
                point = {'X': (maskPoints[j][i][0][0] * downsample) + offset['X'], 'Y': (maskPoints[j][i][0][1] * downsample) + offset['Y']}
                pointList.append(point)
            pointsList.append(pointList)
    return pointsList

### functions for building an xml tree of annotations ###
def xml_create(): # create new xml tree
    # create new xml Tree - Annotations
    Annotations = ET.Element('Annotations', attrib={'MicronsPerPixel': '0.252000'})
    return Annotations

def xml_add_annotation(Annotations, annotationID=None): # add new annotation
    # add new Annotation to Annotations
    # defualts to new annotationID
    if annotationID == None: # not specified
        annotationID = len(Annotations.findall('Annotation')) + 1
    Annotation = ET.SubElement(Annotations, 'Annotation', attrib={'Type': '4', 'Visible': '1', 'ReadOnly': '0', 'Incremental': '0', 'LineColorReadOnly': '0', 'LineColor': str(xml_color[annotationID-1]), 'Id': str(annotationID), 'NameReadOnly': '0'})
    Regions = ET.SubElement(Annotation, 'Regions')
    return Annotations

def xml_add_region(Annotations, pointList,args, annotationID=-1, regionID=None): # add new region to annotation
    # add new Region to Annotation
    # defualts to last annotationID and new regionID
    Annotation = Annotations.find("Annotation[@Id='" + str(annotationID) + "']")
    Regions = Annotation.find('Regions')
    if regionID == None: # not specified
        regionID = len(Regions.findall('Region')) + 1
    Region = ET.SubElement(Regions, 'Region', attrib={'NegativeROA': '0', 'ImageFocus': '-1', 'DisplayId': '1', 'InputRegionId': '0', 'Analyze': '0', 'Type': '0', 'Id': str(regionID)})
    Vertices = ET.SubElement(Region, 'Vertices')
    for point in pointList: # add new Vertex
        ET.SubElement(Vertices, 'Vertex', attrib={'X': str(point['X']*args.approx_downsample), 'Y': str(point['Y']*args.approx_downsample), 'Z': '0'})
    # add connecting point
    ET.SubElement(Vertices, 'Vertex', attrib={'X': str(pointList[0]['X']*args.approx_downsample), 'Y': str(pointList[0]['Y']*args.approx_downsample), 'Z': '0'})
    return Annotations

def xml_save(Annotations, filename):
    xml_data = ET.tostring(Annotations, pretty_print=True)
    #xml_data = Annotations.toprettyxml()
    f = open(filename, 'wb')
    f.write(xml_data)
    f.close()

def read_xml(filename):
    # import xml file
    tree = ET.parse(filename)
    root = tree.getroot()
