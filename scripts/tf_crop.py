import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import pandas as pd
import numpy as np
import random
import glob
import time

source = '/mnt/data/lossless_val_04102022'
filenames = glob.glob(os.path.join(source, '**', "*.*g"), recursive=True)
#file_shuffled_list = random.sample(filenames, len(filenames)//4)

dirname = 'crops'
error_counter = 0
counter = 0
spoofs = []
pths = []

final_destination = source+'_'+dirname

os.makedirs(os.path.join(final_destination,'0'), exist_ok=True)
os.makedirs(os.path.join(final_destination,'1'), exist_ok=True)
start_time = time.time()
for filename in filenames:
    counter = counter + 1
    print(counter)

    try:
        name = filename.split('/')[-1]
        if filename.split('/')[4] == '0':
            spoof = '0'
        elif filename.split('/')[4] == '1':
            spoof = '1'

        image = tf.io.read_file(filename)
        image = tf.io.decode_image(image, channels=3)
        H, W, Ch = np.shape(image)
        if W > H:
            center_image = image[int(0.1*H):int(0.9*H), int(0.2*W):int(0.8*W)]
        else:
            center_image = image[int(0.2*H):int(0.8*H), int(0.1*W):int(0.9*W)]

        for i in range(10):
            # Crop
            batch1_crop = tf.image.random_crop(center_image, size=(128, 128, 3))
            img_png = tf.io.encode_png(batch1_crop, compression=-1, name=None)
            img_pth = '{}/{}/'.format(final_destination, spoof)+name[:-4]+'_{}.png'.format(i)
            tf.io.write_file(img_pth, img_png, name=None)

            # pd df
            pths.append(img_pth)
            spoofs.append(spoof)

    except:
        error_counter = error_counter + 1
        print(filename.split('/'))
        continue

df = pd.DataFrame({'file': pths})
df['spoof'] = spoofs
df.to_csv('{}_{}.csv'.format(source, dirname), index=False)

print('error_counter = ', error_counter)

print('time spent = ', time.time() - start_time)