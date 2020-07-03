import numpy as np
import lxml.etree as ET

def get_slide_bounds(file):

    tree = ET.parse(file)
    root = tree.getroot()

    # get all vertex attributes (points)
    X = []
    Y = []

    for Vertex in root.findall("./Annotation[@Id='1']/Regions/Region[@Id='1']/Vertices/Vertex"):
        # make array of points
        X.append(int(float(Vertex.attrib['X'])))
        Y.append(int(float(Vertex.attrib['Y'])))

    return X,Y
