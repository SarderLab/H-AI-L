### H-AI-L (Human-A.I.-Loop) for semantic segmentation of WSI (whole slide images)

A solution for machine learning for pathologists (easy semantic segmentation of WSI)
This is a pipeline code for iterative training of the DeepLab v2 semantic segmentation network for adaption to WSI. 
Currently this code supports annotation and viewing of WSI boundaries using Aperio ImageScope 
(https://www.leicabiosystems.com/digital-pathology/manage/aperio-imagescope/). 
This is currently limited to files formats which have openslide-python support (.tif support is provided via PIL).

NOTE: currently the only supported ImageScope  annotation tools are the freehand select, and rectangle tools.

Created by [Brendon Lutnick](brendonl@buffalo.edu) and [Brandon Ginley](bgginley@buffalo.edu) at SUNY Buffalo.

This code runs using both python2 and python3, and was tested using ubuntu 16.04

### Dependencies (we think we got them all):

  - Tensorflow        (https://www.tensorflow.org/)
  - OpenSlide Python  (https://openslide.org/)
  - openslide-tools   (https://openslide.org/)
  - OpenCV Python     (http://opencv-python-tutroals.readthedocs.io/en/latest/index.html)
  - NumPy             (http://www.numpy.org/)
  - imgaug            (https://github.com/aleju/imgaug)
  - lxml              (https://lxml.de/)
  - skimage           (http://scikit-image.org/docs/dev/api/skimage.html)
  - matplotlib        (https://matplotlib.org/)
  - imageio           (https://pypi.org/project/imageio/)
  - joblib            (https://pythonhosted.org/joblib/)
  - TkInter           (https://wiki.python.org/moin/TkInter)
  - PIL               (https://pillow.readthedocs.io/en/latest/)

### Usage:

    The code is run by using: segmentation_school.py
    to run this code you must be in the H-AI-L directory where it is contained.

    new project:
        - run segmentation school with the [--project] flag set to the desired name and the [--option] flag set to [new]
        - add new WSI and xml annotations to the 'TRAINING_data/0' folder located in the project folder created

    training:
        - annotate the newly added data / correct the predicted xml files
        - run segmentation school with the [--project] flag set to the desired name and the [--option] flag set to [train]

    prediction:
        - place new unannotated WSI in the newest subfolder of the 'TRAINING_data' folder located in the project folder  
        - run segmentation school with the [--project] flag set to the desired name and the [--option] flag set to [predict]

    validation:
        - place holdout WSI and corresponding xml annotations in the 'HOLDOUT_data' folder located in the project folder
        - run segmentation school with the [--project] flag set to the desired name and the [--option] flag set to [validate]

    The network hyperparameters can be adjusted using the flags provided in segmentation_school.py  

    For non-sparse segmentation of WSI set the [--one_network] flag to 'True'
    this will use only the high resolution (HR) network for training and prediction
