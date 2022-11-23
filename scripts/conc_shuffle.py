import pandas as pd

def conc_shuffle_train():
    df0 = pd.read_csv('/mnt/data/lossless_train_04102022_crops0.csv', sep=',')
    df1 = pd.read_csv('/mnt/data/lossless_train_04102022_crops1.csv', sep=',')

    print(df0)
    print(df1)


    df = pd.concat([df0, df1], ignore_index=True, sort=False)
    df = df.sample(frac=1).reset_index(drop=True)

    print(df)
    df.to_csv('/mnt/data/lossless_train_04102022_crops.csv', sep=',',index=False)
    return df


df = pd.read_csv('/mnt/data/lossless_val_04102022_crops_ord.csv', sep=',')
print(df)
#df = df.sample(frac=1).reset_index(drop=True)
#print(df)
#df.to_csv('/mnt/data/lossless_val_04102022_crops.csv', sep=',',index=False)