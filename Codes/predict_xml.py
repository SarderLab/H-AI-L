import cv2
import numpy as np
import os
import sys
from subprocess import call
import argparse
from joblib import Parallel, delayed
import multiprocessing
from skimage.io import imread, imsave
from skimage.transform import resize
from scipy.ndimage.measurements import label
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects
import lxml.etree as ET
import warnings
from shutil import rmtree

"""
Pipeline code to find gloms from WSI

Call this:
python get_gloms.py --wsi <PATH TO WSI>

"""

def main(args):
    # define folder structure dict
    dirs = {'outDir': args.outDir}
    dirs['xml_save_dir'] = args.xml_save_dir
    dirs['txt_save_dir'] = '/txt_files/'
    dirs['img_save_dir'] = '/img_files/'
    dirs['final_output_dir'] = '/boundaries/'
    dirs['final_boundary_image_dir'] = '/images/'
    dirs['mask_dir'] = '/wsi_mask/'
    dirs['chopped_dir'] = '/originals/'
    dirs['crop_dir'] = '/wsi_crops/'
    dirs['save_outputs'] = args.save_outputs

    # reshape regions calc
    downsample = int(args.downsampleRate**.5)
    region_size = int(args.boxSize*(downsample))
    step = int(region_size*(1-args.overlap_percent))

    # check main directory exists
    make_folder(dirs['outDir'])

    if args.wsi == ' ':
        print('\nPlease specify the whole slide image path\n\nUse flag:')
        print('--wsi <path>\n')

    else:
        # chop wsi
        fileID, test_num_steps = chop_suey(dirs, downsample, region_size, step, args)
        dirs['fileID'] = fileID
        print('Chop SUEY!\n')

        # call DeepLabv2_resnet for prediction
        print('finding Glom locations ...\n')

        make_folder(dirs['outDir'] + fileID + dirs['img_save_dir'] + 'prediction')

        test_data_list = fileID + '_images' + '.txt'

        call(['python3', '/hdd/wsi_fun/Codes/Deeplab-v2--ResNet-101/main.py', '--option', 'predict',
            '--test_data_list', dirs['outDir']+fileID+dirs['txt_save_dir']+test_data_list,
            '--out_dir', dirs['outDir']+fileID+dirs['img_save_dir'], '--test_step', str(args.test_step),
            '--test_num_steps', str(test_num_steps), '--modeldir', args.modeldir,
            '--data_dir', dirs['outDir']+fileID+dirs['img_save_dir'], '--gpu', '1'])

        # un chop
        print('\nreconstructing wsi map ...\n')
        wsiMask = un_suey(dirs=dirs)

        # save hotspots
        if dirs['save_outputs'] == True:
            make_folder(dirs['outDir'] + fileID + dirs['mask_dir'])
            print('saving to: ' + dirs['outDir'] + fileID + dirs['mask_dir'] + fileID  + '.png')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                imsave(dirs['outDir'] + fileID + dirs['mask_dir'] + fileID + '.png', wsiMask)

        # find glom locations in reconstructed map
        print('\ninterpreting prediction map ...')
        test_num_steps, labeledArray, label_offsets = find_suey(wsiMask, dirs, downsample, args)
        print('\n\nthe cropped regions have been saved to: ' + dirs['outDir'] + fileID + dirs['img_save_dir'] + fileID + dirs['crop_dir'])

        # call network 2 to predict Glom boundaries
        print('\ngetting Glom boundaries ...\n')
        make_folder(dirs['outDir'] + fileID + dirs['final_output_dir'] + 'prediction')

        test_data_list = fileID + '_crops.txt'

        call(['python3', '/hdd/wsi_fun/Codes/Deeplab-v2--ResNet-101/main.py', '--option', 'predict',
            '--test_data_list', dirs['outDir']+fileID+dirs['txt_save_dir']+test_data_list,
            '--out_dir', dirs['outDir']+fileID+dirs['final_output_dir'], '--test_step', str(args.test_step_2),
            '--test_num_steps', str(test_num_steps), '--modeldir', args.modeldir_2,
            '--data_dir', dirs['outDir']+fileID+dirs['img_save_dir'], '--gpu', '1'])

        print('\nsaving final glom images ...')
        print('\nworking on:')


        crop_suey(label_offsets, dirs, args)

        # clean up
        if dirs['save_outputs'] == False:
            print('cleaning up')
            rmtree(dirs['outDir']+fileID)

        print('\nall done.\n')

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
    return i + 1

def chop_suey(dirs, downsample, region_size, step, args): # chop wsi
    wsi = args.wsi
    print('\nopening: ' + wsi)
    basename = os.path.splitext(wsi)[0]

    global slide
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

    print('saving region:')
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(chop_wsi)(yStart=i, xStart=j, f_name=f_name, f2_name=f2_name, dirs=dirs, downsample=downsample, region_size=region_size, args=args) for i in index_y for j in index_x)

    test_num_steps = file_len(dirs['outDir'] + fileID + dirs['txt_save_dir'] + fileID + '_images' + ".txt")
    print('\n\n' + str(test_num_steps) +' image regions chopped')

    return fileID, test_num_steps

def chop_wsi(yStart, xStart, f_name, f2_name, dirs, downsample, region_size, args): # perform cutting in parallel
    yEnd = yStart+region_size
	#print(yEnd)
    xEnd = xStart+region_size
	#print(xEnd)
    xLen=xEnd-xStart
    yLen=yEnd-yStart

    subsect= np.array(slide.read_region((xStart,yStart),0,(xLen,yLen)))
    subsect=subsect[:,:,:3]
    grayImage=cv2.cvtColor(subsect,cv2.COLOR_BGR2GRAY)
    np.place(grayImage,grayImage==0, 255)
    whiteRatio=(np.sum(grayImage)/(grayImage.size*255))

    if whiteRatio < args.whiteMax:
		#print(whiteRatio)
        imageIter = str(xStart)+str(yStart)

        f = open(f_name, 'a+')
        f2 = open(f2_name, 'a+')

        # append txt file
        f.write(imageIter + ':' + str(xStart/downsample) + ':' + str(xEnd/downsample)
            + ':' + str(yStart/downsample) + ':' + str(yEnd/downsample) + '\n')

		# resize images ans masks
        c=(subsect.shape)
        s1=int(c[0]/(args.downsampleRate**.5))
        s2=int(c[1]/(args.downsampleRate**.5))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            subsect=resize(subsect,(s1,s2), mode='constant')

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

def un_suey(dirs): # reconstruct wsi from predicted masks
    txtFile = dirs['fileID'] + '.txt'

    # read txt file
    f = open(dirs['outDir'] + dirs['fileID'] + dirs['txt_save_dir'] + txtFile, 'r')
    lines = f.readlines()
    f.close()
    lines = np.array(lines)

    # get wsi size
    xDim = np.int32((lines[1].split(': ')[1]).split('\n')[0])
    yDim = np.int32((lines[2].split(': ')[1]).split('\n')[0])
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
        xStart = np.int32(region[1])
        #print('xStart: ' + str(xStart))
        xStop = np.int32(region[2])
        #print('xStop: ' + str(xStop))
        yStart = np.int32(region[3])
        #print('yStart: ' + str(yStart))
        yStop = np.int32(region[4])
        #print('yStop: ' + str(yStop))

        # populate wsiMask with max
        #print(np.shape(wsiMask))
        wsiMask[yStart:yStop, xStart:xStop] = np.maximum(wsiMask[yStart:yStop, xStart:xStop], mask)
        #wsiMask[yStart:yStop, xStart:xStop] = np.ones([yStop-yStart, xStop-xStart])

    return wsiMask

def find_suey(wsiMask, dirs, downsample, args): # locates the detected glom regions in the reconstructed wsi mask
    # clean up mask
    print('   removing small objects')
    cleanMask = remove_small_objects(wsiMask.astype(bool), args.min_size)
    print('   separating Glom objects\n')
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
    for region_iter in range(1, num_features):
        label_offset = crop_region(region_iter=region_iter, labeledArray=labeledArray, f_name=f_name, dirs=dirs, downsample=downsample, args=args)
        label_offsets.append(label_offset)

    test_num_steps = file_len(dirs['outDir'] + dirs['fileID'] + dirs['txt_save_dir'] + dirs['fileID'] + '_crops' + ".txt")
    return test_num_steps, labeledArray, label_offsets

def crop_region(region_iter, labeledArray, f_name, dirs, downsample, args): # crop selected region from wsi and save // location defined by labeledArray
    # get list of locations for pixels == region_iter
    mask_region = np.argwhere(labeledArray == region_iter)
    # calculate the region bounds
    yStart = min(mask_region[:,0]) * downsample
    yLen = (max(mask_region[:,0]) * downsample) - yStart
    xStart = min(mask_region[:,1]) * downsample
    xLen = (max(mask_region[:,1]) * downsample) - xStart

    region = np.array(slide.read_region((xStart,yStart),0,(xLen,yLen)))

    # print output
    sys.stdout.write('   <' + str(region_iter) + '>   ')
    sys.stdout.flush()
    restart_line()

    # write image path to text file
    f = open(f_name, 'a+')
    f.write(dirs['crop_dir'] + dirs['fileID'] + str(region_iter) + args.imBoxExt + '\n')
    f.close()

    # save image region
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imsave(dirs['outDir'] + dirs['fileID'] + dirs['img_save_dir'] + dirs['crop_dir'] + dirs['fileID'] + str(region_iter) + args.imBoxExt, region)
    label_offset = {'Y': yStart, 'X': xStart}
    return label_offset


def crop_suey(label_offsets, dirs, args):
    txtFile = dirs['fileID'] + '_crops.txt'

    # read txt file with img paths
    f = open(dirs['outDir'] + dirs['fileID'] + dirs['txt_save_dir'] + txtFile, 'r')
    lines = f.readlines()
    f.close()
    lines = np.array(lines)

    make_folder(dirs['outDir'] + dirs['fileID'] + dirs['final_output_dir'] + dirs['final_boundary_image_dir'][1:])

    # make xml
    Annotations = xml_create()
    # add annotation
    Annotations = xml_add_annotation(Annotations=Annotations)

    for line in range(0, np.size(lines)):
        image_path = lines[line].split('\n')[0]

        # get glom and corresponding mask
        file_name = (image_path.split('.')[0]).split(dirs['crop_dir'])[1]
        glom_image = imread(dirs['outDir'] + dirs['fileID'] + dirs['img_save_dir'] + image_path[1:])
        mask_image = imread(dirs['outDir'] + dirs['fileID'] + dirs['final_output_dir'] + 'prediction/'
            + file_name + '_mask.png')

        # print output
        sys.stdout.write('   <' + file_name + '>   ')
        sys.stdout.flush()
        restart_line()

        # add mask to xml
        label_offset = label_offsets[line]
        pointsList = get_contour_points(mask_image, offset=label_offset)
        for i in range(np.shape(pointsList)[0]):
            pointList = pointsList[i]
            Annotations = xml_add_region(Annotations=Annotations, pointList=pointList)

        # save mask images
        if dirs['save_outputs'] == True:
            if np.sum(mask_image) != 0:
                # remove background in images
                for i in range(3):
                    glom_image[:,:,i] = glom_image[:,:,i] * (mask_image * ((1-args.bg_intensity)) + args.bg_intensity)

                    # save resulting image
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        imsave(dirs['outDir'] + dirs['fileID'] + dirs['final_output_dir'] + dirs['final_boundary_image_dir'][1:] + file_name + '_glom' + args.finalImgExt, glom_image)

    # save xml
    xml_save(Annotations=Annotations, filename=dirs['xml_save_dir']+'/'+dirs['fileID']+'.xml')

def get_contour_points(mask, offset={'X': 0,'Y': 0}):
    # returns a dict pointList with point 'X' and 'Y' values
    # input greyscale binary image
    maskPoints, contours = cv2.findContours(np.array(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pointsList = []
    for j in range(np.shape(maskPoints)[0]):
        pointList = []
        for i in range(np.shape(maskPoints[j])[0]):
            point = {'X': maskPoints[j][i][0][0] + offset['X'], 'Y': maskPoints[j][i][0][1] + offset['Y']}
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
    Annotation = ET.SubElement(Annotations, 'Annotation', attrib={'Type': '4', 'Visible': '1', 'ReadOnly': '0', 'Incremental': '0', 'LineColorReadOnly': '0', 'LineColor': '65280', 'Id': str(annotationID), 'NameReadOnly': '0'})
    Regions = ET.SubElement(Annotation, 'Regions')
    return Annotations

def xml_add_region(Annotations, pointList, annotationID=-1, regionID=None): # add new region to annotation
    # add new Region to Annotation
    # defualts to last annotationID and new regionID
    Annotation = Annotations[annotationID]
    Regions = Annotation.find('Regions')
    if regionID == None: # not specified
        regionID = len(Regions.findall('Region')) + 1
    Region = ET.SubElement(Regions, 'Region', attrib={'NegativeROA': '0', 'ImageFocus': '-1', 'DisplayId': '1', 'InputRegionId': '0', 'Analyze': '0', 'Type': '0', 'Id': str(regionID)})
    Vertices = ET.SubElement(Region, 'Vertices')
    for point in pointList: # add new Vertex
        ET.SubElement(Vertices, 'Vertex', attrib={'X': str(point['X']), 'Y': str(point['Y']), 'Z': '0'})
    # add connecting point
    ET.SubElement(Vertices, 'Vertex', attrib={'X': str(pointList[0]['X']), 'Y': str(pointList[0]['Y']), 'Z': '0'})
    return Annotations

def xml_save(Annotations, filename):
    xml_data = ET.tostring(Annotations, pretty_print=True)
    #xml_data = Annotations.toprettyxml()
    f = open(filename, 'w')
    f.write(xml_data)
    f.close()

def read_xml(filename):
    # import xml file
    tree = ET.parse(filename)
    root = tree.getroot()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--wsi', dest='wsi', default=' ',type=str,
        help='Please specify the whole slide image path')

    ### Params for saving results ###
    parser.add_argument('--outDir', dest='outDir', default='/hdd/IterativeAnnotation/Gloms/' ,type=str,
    help='output directory')
    parser.add_argument('--xml_save_dir', dest='xml_save_dir', default='/hdd/IterativeAnnotation/Gloms/' ,type=str,
    help='directory where xml file is saved')
    parser.add_argument('--save_outputs', dest='save_outputs', default=False ,type=bool,
    help='save outputs from chopping etc. [final image masks]')
    parser.add_argument('--bg_intensity', dest='bg_intensity', default=.5 ,type=float,
    help='if displaying output classifications [save_outputs = True] background color [0-1]')

    ### Params for cutting wsi ###
    parser.add_argument('--overlap_percent', dest='overlap_percent', default=0.5 ,type=float,
        help='overlap percentage of blocks [0-1]')
    parser.add_argument('--boxSize', dest='boxSize', default=750 ,type=int,
        help='size of blocks')
    parser.add_argument('--downsampleRate', dest='downsampleRate', default=16 ,type=int,
        help='downsample rate for low rez network')
    parser.add_argument('--imBoxExt', dest='imBoxExt', default='.jpeg' ,type=str,
        help='ext of saved image blocks')
    parser.add_argument('--finalImgExt', dest='finalImgExt', default='.jpeg' ,type=str,
        help='ext of final saved images')
    parser.add_argument('--whiteMax', dest='whiteMax', default=0.9 ,type=float,
        help='exclude white blocks')

    ### Params for network to test with
    parser.add_argument('--modeldir', dest='modeldir', default='/home/wsi_fun/Codes/model1' ,type=str,
        help='prepass model folder')
    parser.add_argument('--test_step', dest='test_step', default=217000 ,type=int,
        help='prepass model iteration')
    parser.add_argument('--modeldir_2', dest='modeldir_2', default='/home/wsi_fun/Codes/model2' ,type=str,
        help='second pass model folder')
    parser.add_argument('--test_step_2', dest='test_step_2', default=234000 ,type=int,
        help='second pass model iteration')

    ### Params for optimizing wsi mask cleanup
    parser.add_argument('--min_size', dest='min_size', default=625 ,type=int,
        help='min size region to be considered after prepass [in pixels]')


    args = parser.parse_args()

    main(args=args)
