import pandas as pd

csv_train = '/mnt/data/lossless_train_04102022_crops.csv'
csv_test = '/mnt/data/lossless_val_04102022_crops.csv'

df = pd.read_csv(csv_test, sep=',')
print(df)