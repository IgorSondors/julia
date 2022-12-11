import sys
import os
import glob
import pandas as pd

def drop_from_list(csv_train):
    files_to_remove = set(['6f8b0342-36b4-4d3f-917b-3445b4e9be70.png',
        '852d0999-0155-4bdb-915c-a2e44d509374.png',
        'a767c60f-828b-4cd2-a5a3-d449e4c33438.png',
        'c9fe59d3-8c50-4f23-ab20-be6f5b7a9288.png',
        'dec5f84d-3c3b-4257-a1c6-3fe7a66e0675.png',
        '6ffdebbf-7d8e-42a2-b6e5-66f5d94c61c1.png',
        '7dc497cb-6f56-4726-bf50-869d047bfcd9.png',
        '46b0ca61-3e49-47a8-8407-32e86c45e6ad.png',
        '65184990-4ec8-470b-9e4a-81f6084d6c98.png',
        '9928c602-93e2-4a38-8fbc-978fbb70ecb4.png',
        '435cee70-eba6-41c0-85ea-44ef5ba8a614.png',
        '0e2fb49c-d12a-405a-9e70-e497502fa26b.png',
        '4dd89c83-56d8-4f06-9824-7cf7ff7a4f19.png',
        '7d7f73b4-d24b-4378-9800-47b3fb35af09.png',
        '36b9c44b-490b-4aff-8800-1ac06111be66.png',
        'd5f0e4a2-76be-4d21-82cf-c81ee9351e4a.png',
        '993e23fb-8e66-48e4-b939-31a8d2a39fac.png',
        'f5842ebb-9347-401a-8c04-63643034141e.png',
        'f7b5727d-8392-4bc2-998a-3bba8865158e.png',
        '1ecd514b-e81e-4d47-9583-dc70b93b7cab.png',
        '4c84f2bf-e05e-422d-8e77-f88238a7e60a.png',
        '2db42fbd-a391-4b8f-b479-d9c4075e5cb9.png',
        '8e5c5c03-832f-4841-ac54-1f5e963ce22c.png',
        '670ee9a1-d115-44b3-8184-bfb03371a92a.png'])

    df_train = pd.read_csv(csv_train, sep=',')
    print(df_train)
    grb_ind = []
    for index, row in df_train.iterrows():
        #print(index)
        img_file = row['file']
        name = img_file.split('/')[-1][:-6]+'.png'
        if name in files_to_remove:
            print(name)
            grb_ind.append(index)
    print(grb_ind)
    df_train.drop(grb_ind, axis=0, inplace=True)
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    print(df_train)
    df_train.to_csv('/mnt/data/camera_spoofing/lossless_train_16072022_crops_full_clear.csv', index=False)

def drop_by_kb(csv_train):
    csv_train = '/mnt/data/camera_spoofing/lossless_train_16072022_crops_10_drop_repeats.csv'
    csv_blacklist = '/mnt/data/camera_spoofing/lossless_train_16072022_blacklist.csv'

    df_train = pd.read_csv(csv_train, sep=',')
    df_blacklist = pd.read_csv(csv_blacklist, sep=',')
    print(df_blacklist)
    grb_ind = []
    files_to_remove = list(df_blacklist['file'])
    print(files_to_remove)

    for index, row in df_train.iterrows():
        #print(index)
        img_file = row['file']
        name = img_file.split('/')[-1][:-6]
        if name in files_to_remove:
            print(name)
            grb_ind.append(index)

    len_df_train_old = len(df_train)
    df_train.drop(grb_ind, axis=0, inplace=True)
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_train.to_csv('/mnt/data/camera_spoofing/lossless_train_16072022_crops_10_drop_repeats_100kb.csv', index=False)

    print(len(df_train))
    print(len_df_train_old)
    print(len_df_train_old - len(df_train))

def add_ugreen(csv_train):
    csv_train = '/mnt/data/camera_spoofing/lossless_train_16072022_crops_10_drop_repeats_100kb.csv'
    csv_train1 = '/mnt/data/camera_spoofing/ugreen_28072022_crops.csv'
    csv_train2 = '/mnt/data/camera_spoofing/ugreen_29072022_crops.csv'

    df_train = pd.read_csv(csv_train, sep=',')
    df_train1 = pd.read_csv(csv_train1, sep=',')
    df_train2 = pd.read_csv(csv_train2, sep=',')

    df_train_all = pd.concat([df_train, df_train1, df_train2], axis=0)
    df_train_all = df_train_all.sample(frac=1).reset_index(drop=True)
    df_train_all.to_csv('/mnt/data/camera_spoofing/lossless_train_16072022_ugreen_crops_10.csv', index=False)

    print(df_train_all)

    print(len(df_train))
    print(len(df_train1))
    print(len(df_train2))
    print(len(df_train_all))