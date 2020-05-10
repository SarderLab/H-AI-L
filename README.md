### H-AI-L (Human-A.I.-Loop) for semantic segmentation of WSI (whole slide images)

A solution for machine learning for pathologists (easy semantic segmentation of WSI)
This is a library of codes for iterative training of the DeepLab v2 semantic segmentation network for WSIs. 
Currently this code supports annotation and viewing of WSI boundaries using Aperio ImageScope 
(https://www.leicabiosystems.com/digital-pathology/manage/aperio-imagescope/). 
This is currently limited to files formats which have openslide-python support (.tif support is provided via PIL).

### The published paper is available here: https://www.nature.com/articles/s42256-019-0018-3
### A preprint version of this work can be found here: https://arxiv.org/abs/1812.07509

NOTE: currently the only supported ImageScope  annotation tools are the freehand select, and rectangle tools.

Created by [Brendon Lutnick](https://github.com/brendonlutnick) and [Brandon Ginley](https://github.com/bgginley) at SUNY Buffalo.

This code runs using python3, and was tested using ubuntu 16.04

### Dependencies (we think we got them all):

  - Tensorflow (1.7.0)           (https://www.tensorflow.org/)
  - OpenSlide Python (1.1.1)     (https://openslide.org/)
  - openslide-tools              (https://openslide.org/)
  - OpenCV Python (3.4.0.12)     (http://opencv-python-tutroals.readthedocs.io/en/latest/index.html)
  - NumPy (1.14.2)               (http://www.numpy.org/)
  - imgaug (0.2.5)               (https://github.com/aleju/imgaug)
  - lxml (3.5.0)                 (https://lxml.de/)
  - skimage (0.15.0)             (http://scikit-image.org/docs/dev/api/skimage.html)
  - matplotlib (2.2.2)           (https://matplotlib.org/)
  - imageio (2.3.0)              (https://pypi.org/project/imageio/)
  - joblib (0.11)                (https://pythonhosted.org/joblib/)
  - TkInter (3.7.2)              (https://wiki.python.org/moin/TkInter)
  - PIL (5.3.0)                  (https://pillow.readthedocs.io/en/latest/)
  
  Before starting a new project, you must download a pretrained checkpoint file containing model parameters
  - available here: https://buffalo.box.com/s/ks7m9lf5h4ilzwyqdnlr9h8xdvd13a5i
  Place this file in the [H-AI-L/Codes/Deeplab_network] folder

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

### Quick start guide
First, download the source codes and unzip them. Change the directory to the unzipped folder and run
  python segmentation_school.py --option new --project project
Where 'project' is the desired name of the directory to contain the new HAIL project.

Next, place training WSIs and XML annotations OR prediction WSIs in the folder your_project_name/TRAINING_data/0/. Place your pre-trained model(s) into the your_project_name/MODELS/0/HR/ (for dense prediction) and/or your_project_name/MODELS/0/LR/ (for sparse prediction) folders.
Then, run
'''
  python segmentation_school.py --option option --project project --encoder_name encoder_name --one_network one_network
'''
Where 'option' is either train or predict,'project' is the directory name you specified before, 'encoder_name' is either res50, res101, or deeplab, and 'one_network' is either true (dense prediction with one network) or false (sparse prediction with cascaded low-high resolution networks).
Please open and refer to the segmentation_school.py script for a full list of possible input parameters and their default values. 
