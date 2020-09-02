import numpy as np
import sys
import lxml.etree as ET
import cv2

def get_num_classes(xml_path):
    # parse xml and get root
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotation_num = 0
    for Annotation in root.findall("./Annotation"): # for all annotations
        annotation_num = annotation_num + 1

    return annotation_num + 1


"""
location (tuple) - (x, y) tuple giving the top left pixel in the level 0 reference frame
size (tuple) - (width, height) tuple giving the region size

"""

def xml_to_mask(xml_path, location, size, downsample_factor=1, verbose=0):
    # parse xml and get root
    
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # calculate region bounds
    bounds = {'x_min' : location[0], 'y_min' : location[1], 'x_max' : location[0] + size[0], 'y_max' : location[1] + size[1]}

    IDs = regions_in_mask(root=root, bounds=bounds, verbose=verbose)

    if verbose != 0:
        print('\nFOUND: ' + str(len(IDs)) + ' regions')

    # find regions in bounds
    Regions = get_vertex_points(root=root, IDs=IDs, verbose=verbose)

    # fill regions and create mask
    mask = Regions_to_mask(Regions=Regions, bounds=bounds, IDs=IDs, downsample_factor=downsample_factor, verbose=verbose)
    if verbose != 0:
        print('done...\n')

    return mask

def restart_line(): # for printing labels in command line
    sys.stdout.write('\r')
    sys.stdout.flush()

def regions_in_mask(root, bounds, verbose=1):
    # find regions to save
    IDs = []

    for Annotation in root.findall("./Annotation"): # for all annotations
        annotationID = Annotation.attrib['Id']

        for Region in Annotation.findall("./*/Region"): # iterate on all region

            if verbose != 0:
                sys.stdout.write('TESTING: ' + 'Annotation: ' + annotationID + '\tRegion: ' + Region.attrib['Id'])
                sys.stdout.flush()
                restart_line()

            for Vertex in Region.findall("./*/Vertex"): # iterate on all vertex in region
                # get points
                x_point = np.int32(np.float64(Vertex.attrib['X']))
                y_point = np.int32(np.float64(Vertex.attrib['Y']))
                # test if points are in bounds
                if bounds['x_min'] <= x_point <= bounds['x_max'] and bounds['y_min'] <= y_point <= bounds['y_max']: # test points in region bounds
                    # save region Id
                    IDs.append({'regionID' : Region.attrib['Id'], 'annotationID' : annotationID})
                    break
    return IDs

def get_vertex_points(root, IDs, verbose=1):
    Regions = []

    for ID in IDs: # for all IDs
        if verbose != 0:
            sys.stdout.write('PARSING: ' + 'Annotation: ' + ID['annotationID'] + '\tRegion: ' + ID['regionID'])
            sys.stdout.flush()
            restart_line()

        # get all vertex attributes (points)
        Vertices = []

        for Vertex in root.findall("./Annotation[@Id='" + ID['annotationID'] + "']/Regions/Region[@Id='" + ID['regionID'] + "']/Vertices/Vertex"):
            # make array of points
            Vertices.append([int(float(Vertex.attrib['X'])), int(float(Vertex.attrib['Y']))])


        Regions.append(np.array(Vertices))

    return Regions

def Regions_to_mask(Regions, bounds, IDs, downsample_factor, verbose=1):
    downsample = int(np.round(downsample_factor**(.5)))

    if verbose !=0:
        print('\nMAKING MASK:')

    if len(Regions) != 0: # regions present
        # get min/max sizes
        min_sizes = np.empty(shape=[2,0], dtype=np.int32)
        max_sizes = np.empty(shape=[2,0], dtype=np.int32)
        for Region in Regions: # fill all regions
            min_bounds = np.reshape((np.amin(Region, axis=0)), (2,1))
            max_bounds = np.reshape((np.amax(Region, axis=0)), (2,1))
            min_sizes = np.append(min_sizes, min_bounds, axis=1)
            max_sizes = np.append(max_sizes, max_bounds, axis=1)
        min_size = np.amin(min_sizes, axis=1)
        max_size = np.amax(max_sizes, axis=1)



        # add to old bounds
        bounds['x_min_pad'] = min(min_size[1], bounds['x_min'])
        bounds['y_min_pad'] = min(min_size[0], bounds['y_min'])
        bounds['x_max_pad'] = max(max_size[1], bounds['x_max'])
        bounds['y_max_pad'] = max(max_size[0], bounds['y_max'])

        # make blank mask
        mask = np.zeros([ int(np.round((bounds['y_max_pad'] - bounds['y_min_pad']) / downsample)), int(np.round((bounds['x_max_pad'] - bounds['x_min_pad']) / downsample)) ], dtype=np.int8)


        # fill mask polygons
        index = 0
        for Region in Regions:
            # reformat Regions
            Region[:,1] = np.int32(np.round((Region[:,1] - bounds['y_min_pad']) / downsample))
            Region[:,0] = np.int32(np.round((Region[:,0] - bounds['x_min_pad']) / downsample))
            # get annotation ID for mask color
            ID = IDs[index]
            cv2.fillPoly(mask, [Region], int(ID['annotationID']))
            index = index + 1

        # reshape mask
        x_start = np.int32(np.round((bounds['x_min'] - bounds['x_min_pad']) / downsample))
        y_start = np.int32(np.round((bounds['y_min'] - bounds['y_min_pad']) / downsample))
        x_stop = np.int32(np.round((bounds['x_max'] - bounds['x_min_pad']) / downsample))
        y_stop = np.int32(np.round((bounds['y_max'] - bounds['y_min_pad']) / downsample))
        # pull center mask region
        mask = mask[ y_start:y_stop, x_start:x_stop ]

    else: # no Regions
        mask = np.zeros([ int(np.round((bounds['y_max'] - bounds['y_min']) / downsample)), int(np.round((bounds['x_max'] - bounds['x_min']) / downsample)) ])

    return mask
