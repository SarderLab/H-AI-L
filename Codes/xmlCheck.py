from xml_to_mask import xml_to_mask
from getWsi import getWsi
from matplotlib import pyplot as plt

slide=getWsi('/hdd/bg/HAIL2/DeepZoomPrediction/TRAINING_data/0/52483.svs')
[d1,d2]=slide.dimensions
x='/hdd/bg/HAIL2/DeepZoomPrediction/TRAINING_data/0/52483.xml'
wsiMask=xml_to_mask(x,(0,0),(d1,d2),16,0)

plt.imshow(wsiMask*255)
plt.show()
