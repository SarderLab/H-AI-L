from glob import glob
import os

def check_missing(start, to, ext):
    files = glob(start)
    total = len(files)
    tot = 0
    for iter, file in enumerate(files):
        base = file.split('/')[-1]
        base = base.split('.')[0]
        print('\r[{} of {}] | {}'.format(iter, total, file), end='')
        if not os.path.isfile('{}/{}{}'.format(to, base, ext)):
            tot += 1
            os.remove(file)

    return tot

print('\n1 of 4')
tot = check_missing('./Permanent/LR/masks/*.png', './Permanent/LR/regions/', '.jpeg')
print('\n\tremoved: {}'.format(tot))

print('\n2 of 4')
tot = check_missing('./Permanent/LR/regions/*.jpeg', './Permanent/LR/masks/', '.png')
print('\n\tremoved: {}'.format(tot))

print('\n3 of 4')
tot = check_missing('./Permanent/HR/masks/*.png', './Permanent/HR/regions/', '.jpeg')
print('\n\tremoved: {}'.format(tot))

print('\n4 of 4')
tot = check_missing('./Permanent/HR/regions/*.jpeg', './Permanent/HR/masks/', '.png')
print('\n\tremoved: {}'.format(tot))
