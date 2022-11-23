import pandas as pd
import os

csv_train = '/mnt/data/lossless_train_04102022_crops_sizes.csv'
csv_test = '/mnt/data/lossless_val_04102022_crops_sizes.csv'

def write_sizes(csv):
    df = pd.read_csv(csv, sep=',')
    print(df)

    for index, row in df.iterrows():
        file = row['file']
        size = os.path.getsize(file)
        df.loc[index, 'size'] = size

    df.to_csv('/mnt/data/lossless_train_04102022_crops_sizes.csv', sep=',',index=False)
    return df

#write_sizes(csv_train)