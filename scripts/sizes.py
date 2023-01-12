import os
import glob
import pandas as pd
source = '/home/ubuntu/camera_spoofing/lossless_unified/lossless_val_10012023'
filenames = glob.glob(os.path.join(source, '**', "*.*g"), recursive=True)

small_counter_0 = 0
small_counter_1 = 0
sizes = 0
sizes_list = []
blacklist = []
for filename in filenames:
    if '0' in filename.split('/'):
        spoof = '0'
    elif '1' in filename.split('/'):
        spoof = '1'
    name = filename.split('/')[-1][:-(1+len(filename.split('.')[-1]))]
    size = os.path.getsize(filename)
    sizes = sizes + size
    sizes_list.append(size)
    if filename.split('.')[-1] != 'png':
        size = 0
    if size < 100000:
        if spoof == '0':
            small_counter_0 = small_counter_0 + 1
            print(filename, '-->', size)
        elif spoof == '1':
            small_counter_1 = small_counter_1 + 1
            print(filename, '-->', size)
        blacklist.append(name)    
print('small_counter_0 = ', small_counter_0)
print('small_counter_1 = ', small_counter_1)
print('sizes total = ', sizes)
print('sizes mean = ', sizes/len(filenames))
print('min(sizes_list) = ', min(sizes_list))
print('max(sizes_list) = ', max(sizes_list))
print(len(filenames))

df = pd.DataFrame({'file': blacklist})
df.to_csv('{}_blacklist.csv'.format(source), index=False)

print(df)