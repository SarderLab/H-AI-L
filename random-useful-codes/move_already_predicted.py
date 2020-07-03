import os
from glob import glob

dir = 'Predicted_XMLs/'
print('\n')

xmls = glob('{}*.xml'.format(dir))
for xml in xmls:
    base = xml.split('.xml')[-2]
    base = base.split(dir)[-1]


    try:
        wsi = glob('{}.*'.format(base))[0]
        print('moving: [{}]'.format(base))
        os.rename(wsi, '{}/{}'.format(dir,wsi))
    except: pass
