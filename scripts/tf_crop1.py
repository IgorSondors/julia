import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import pandas as pd
import numpy as np
#import random
import glob

source = '/home/ubuntu/camera_spoofing/lossless_unified/train'
final_destination = '/home/ubuntu/camera_spoofing/lossless_unified/train_10random_crops'
os.makedirs(os.path.join(final_destination,'0'), exist_ok=True)
os.makedirs(os.path.join(final_destination,'1'), exist_ok=True)

filenames = glob.glob(os.path.join(source, '**', "*.*g"), recursive=True)
#file_shuffled_list = random.sample(filenames, len(filenames)//4)

print(len(filenames))
error_counter = 0
counter = 0
error_pths = []
pths = []
spoof = '1'

for filename in filenames:   
    try:
        if spoof in filename.split('/'):
            counter = counter + 1
            print(counter)

            name = filename.split('/')[-1]
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
        else:
            continue

    except:
        error_counter = error_counter + 1
        error_pths.append(filename)
        print(filename.split('/'))
        continue

spoofs = [spoof for i in range(len(pths))]
df = pd.DataFrame({'file': pths})
df['spoof'] = spoofs
df.to_csv('/home/ubuntu/camera_spoofing/lossless_unified/train_10random_crops1.csv', index=False)

df_error = pd.DataFrame({'error_file': error_pths})
df_error.to_csv('/home/ubuntu/camera_spoofing/lossless_unified/train_errors1.csv', index=False)

print('files_for_crop_counter = ', counter)
print('crops_counter = ', len(pths))

print('error_files_counter = ', len(error_pths))
print('error_counter = ', error_counter)
