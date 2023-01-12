import pandas as pd

def conc_shuffle_train(csv0, csv1, dst):
    df0 = pd.read_csv(csv0, sep=',')
    df1 = pd.read_csv(csv1, sep=',')

    print(df0)
    print(df1)


    df = pd.concat([df0, df1], ignore_index=True, sort=False)
    df = df.sample(frac=1).reset_index(drop=True)

    print(df)
    df.to_csv(dst, sep=',',index=False)
    return df


df = pd.read_csv('/home/ubuntu/camera_spoofing/lossless_unified/val_10random_crops.csv', sep=',')

print(df)
print(len(df))


df = df.sample(frac=1).reset_index(drop=True)
print(df)
print(len(df))

df.to_csv('/home/ubuntu/camera_spoofing/lossless_unified/lossless_val_10012023_10random_crops.csv', sep=',',index=False)

#csv0 = '/home/ubuntu/camera_spoofing/lossless_unified/train_10random_crops0.csv'
#csv1 = '/home/ubuntu/camera_spoofing/lossless_unified/train_10random_crops1.csv'
#dst = '/home/ubuntu/camera_spoofing/lossless_unified/lossless_train_10012023_10random_crops.csv'
#df = conc_shuffle_train(csv0, csv1, dst)