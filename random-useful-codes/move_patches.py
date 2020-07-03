from glob import glob
import os

def move_patches(start, to):
    files = glob(start)
    total = len(files)
    for iter, file in enumerate(files):
        base = file.split('/')[-1]
        print('\r[{} of {}] | {}'.format(iter, total, file), end='')
        # print(file, '{}/{}'.format(to, base))
        os.rename(file, '{}/{}'.format(to, base))

print('\n1 of 8')
move_patches('./TempLR/Augment/masks/*.png', './Permanent/LR/masks/')
print('\n2 of 8')
move_patches('./TempLR/Augment/regions/*.jpeg', './Permanent/LR/regions/')

print('\n3 of 8')
move_patches('./TempHR/Augment/masks/*.png', './Permanent/HR/masks/')
print('\n4 of 8')
move_patches('./TempHR/Augment/regions/*.jpeg', './Permanent/HR/regions/')

print('\n5 of 8')
move_patches('./TempLR/masks/*.png', './Permanent/LR/masks/')
print('\n6 of 8')
move_patches('./TempLR/regions/*.jpeg', './Permanent/LR/regions/')

print('\n7 of 8')
move_patches('./TempHR/masks/*.png', './Permanent/HR/masks/')
print('\n8 of 8')
move_patches('./TempHR/regions/*.jpeg', './Permanent/HR/regions/')

print('\n\nall done...')
