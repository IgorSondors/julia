import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D, Conv2D, Input, Flatten, concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, AveragePooling2D, multiply, Lambda
from tensorflow.keras.layers import Layer
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler, ReduceLROnPlateau

import tensorflow as tf

import pandas as pd
import numpy as np
from math import sqrt
from math import exp

import sys
sys.path.append('/home/yandex/igor/julia/arcface_more_classes')

from arcface_tools import *


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def dct_2d(
        feature_map,
        norm=None # can also be 'ortho'
):
    X1 = tf.signal.dct(feature_map, type=2, norm=norm)
    X1_t = tf.transpose(X1, perm=[0, 1, 3, 2])
    X2 = tf.signal.dct(X1_t, type=2, norm=norm)
    X2_t = tf.transpose(X2, perm=[0, 1, 3, 2])
    return X2_t
        
def white_norm(input):
    return (input - tf.constant(127.5)) / 128.0

def norm(input):
    return (input) / 255.0

def BRA(input):
    bn = BatchNormalization()(input)
    activation = Activation('swish')(bn)
    return AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(activation)


def SE_BLOCK(input, using_SE=True, r_factor=2):
    if not using_SE:
        return input
    channel_nums = input.get_shape()[-1]
    ga_pooling = GlobalAveragePooling2D()(input)
    fc1 = Dense(channel_nums // r_factor, activation='swish')(ga_pooling)
    scale = Dense(channel_nums, activation='sigmoid')(fc1)
    return multiply([scale, input])

def fft2d_function(x, dtype = "complex64"):
    #(None, 128, 128, 1)
    #x=tf.expand_dims(x, -1)
    x = tf.transpose(x, perm=[0, 3, 1, 2])#perm = [2, 0, 1])
    x = tf.cast(x, dtype)
    x_f = tf.signal.fft2d(x)
    x_f = tf.transpose(x_f, perm=[0, 2, 3, 1])#perm = [1, 2, 0])
    real_x_f, imag_x_f = tf.math.real(x_f), tf.math.imag(x_f)
    return real_x_f, imag_x_f

def fft3d_function(x, dtype = "complex64"):
    x = tf.transpose(x, perm=[0, 3, 1, 2])
    x = tf.cast(x, dtype)
    x_f = tf.signal.fft3d(x)
    x_f = tf.transpose(x_f, perm=[0, 2, 3, 1])
    real_x_f, imag_x_f = tf.math.real(x_f), tf.math.imag(x_f)
    return real_x_f, imag_x_f

def jpeg_compression(x, dtype = "uint8"):
    x = tf.cast(x, dtype)

    uint8_img_jpeg = tf.io.encode_jpeg(
        x,
        format='',
        quality=99,
        progressive=False,
        optimize_size=False,
        chroma_downsampling=True,
        density_unit='in',
        x_density=300,
        y_density=300,
        xmp_metadata='',
        name=None
    )
    uint8_img_jpeg = tf.io.decode_image(uint8_img_jpeg, channels=3)
    uint8_img_jpeg = tf.reshape(uint8_img_jpeg, [128, 128, 3])
    return tf.cast(uint8_img_jpeg, tf.float32)

def apply_on_each_element(input_image):
    return tf.map_fn(jpeg_compression, input_image)
    
def highpass_on_each_element(input_image):
    return tf.map_fn(tf_highpass_filter, input_image)

def build_shared_plain_network(input_image, mode, using_SE=True):
    """
    Build plain model
    From https://github.com/StevenBanama/C3AE
    """


    if mode=='rgb':
        wn = Lambda(white_norm, name="white_norm_{}".format(mode))(input_image)
        print('np.shape(wn) = ', np.shape(wn))

        conv1 = Conv2D(32, (3, 3), use_bias=False)(wn)

    elif mode=='dct':
        wn = Lambda(white_norm, name="white_norm_{}".format(mode))(input_image)
        print('np.shape(wn) = ', np.shape(wn))

        dct = Lambda(dct_2d, name='dct')(wn)
        print('np.shape(dct) = ', np.shape(dct))
        conv1 = Conv2D(32, (3, 3), use_bias=False)(dct)

    elif mode=='highpass_filter':
        #norm_img = Lambda(norm, name="norm_{}".format(mode))(input_image)
        #rint('np.shape(wn) = ', np.shape(wn))

        highpass_filter = Lambda(highpass_on_each_element, name='highpass_filter')(input_image)
        print('np.shape(norm) = ', np.shape(highpass_filter))
        conv1 = Conv2D(32, (3, 3), use_bias=False)(highpass_filter)

    elif mode=='fft3d':
        wn = Lambda(white_norm, name="white_norm_{}".format(mode))(input_image)
        print('np.shape(wn) = ', np.shape(wn))

        fft_r, fft_i = Lambda(fft3d_function, name='fft')(wn)
        fft_r32 = tf.cast(fft_r, tf.float32)
        fft_i32 = tf.cast(fft_i, tf.float32)
        fft = tf.concat([fft_r32, fft_i32], axis=3)
        print('np.shape(fft) = ', np.shape(fft))
        conv1 = Conv2D(32, (3, 3), use_bias=False)(fft)

    elif mode=='fft3d_r':
        wn = Lambda(white_norm, name="white_norm_{}".format(mode))(input_image)
        print('np.shape(wn) = ', np.shape(wn))

        fft_r, fft_i = Lambda(fft3d_function, name='fft3d_r')(wn)
        fft_r32 = tf.cast(fft_r, tf.float32)
        print('np.shape(fft_r32) = ', np.shape(fft_r32))
        conv1 = Conv2D(32, (3, 3), use_bias=False)(fft_r32)

    elif mode=='fft3d_im':
        wn = Lambda(white_norm, name="white_norm_{}".format(mode))(input_image)
        print('np.shape(wn) = ', np.shape(wn))

        fft_r, fft_i = Lambda(fft3d_function, name='fft3d_im')(wn)
        fft_i32 = tf.cast(fft_i, tf.float32)
        print('np.shape(fft_i32) = ', np.shape(fft_i32))
        conv1 = Conv2D(32, (3, 3), use_bias=False)(fft_i32)

    elif mode=='jpeg':
        print('np.shape(input_image) = ', np.shape(input_image))

        img_jpeg = Lambda(apply_on_each_element, name="img_jpeg")(input_image)
        print('np.shape(img_jpeg) = ', np.shape(img_jpeg))

        wn = Lambda(white_norm, name="white_norm_{}".format(mode))(img_jpeg)
        print('np.shape(wn) = ', np.shape(wn))

        dct = Lambda(dct_2d, name="dct_{}".format(mode))(wn)
        print('np.shape(dct) = ', np.shape(dct))

        conv1 = Conv2D(32, (3, 3), use_bias=False)(dct)

    block1 = BRA(input=conv1)
    block1 = SE_BLOCK(input=block1, using_SE=using_SE)

    conv2 = Conv2D(32, (3, 3), padding="valid", strides=1, name="conv2_{}".format(mode))(block1)
    block2 = BRA(conv2)
    block2 = SE_BLOCK(block2, using_SE)

    conv3 = Conv2D(32, (3, 3), padding="valid", strides=1, name="conv3_{}".format(mode))(block2)
    block3 = BRA(conv3)
    block3 = SE_BLOCK(block3, using_SE)

    conv4 = Conv2D(32, (3, 3), use_bias=False, name="conv4_{}".format(mode))(block3)
    block4 = BatchNormalization()(conv4)
    block4 = Activation(activation='swish')(block4)
    block4 = SE_BLOCK(block4, using_SE)

    conv5 = Conv2D(32, (1, 1), padding="valid", strides=1, name="conv5_{}".format(mode))(block4)
    conv5 = SE_BLOCK(conv5, using_SE)

    flat_conv = Flatten()(conv5)

    pmodel = Model(inputs=input_image, outputs=[flat_conv])
    return pmodel
    
def create_model(img_size):

    input_tensor = Input(shape=(img_size, img_size, 1))

    base_model_dct = build_shared_plain_network(input_tensor, mode='highpass_filter')
    op_dct = Dropout(.5)(base_model_dct.output)
    x_dct = Dense(32, activation='swish')(op_dct)

    output_tensor = Dense(1, activation='sigmoid')(x_dct)
    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model


class DataGenerator_highpass(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, img_paths, img_labels, img_size, batch_size, n_classes=2, shuffle=False):
        'Initialization'
        self.batch_size = batch_size
        self.batch_size = img_size
        self.img_labels = img_labels
        self.img_paths = img_paths
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.img_paths) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        img_paths_temp = [self.img_paths[k] for k in indexes]
        img_labels_temp = [self.img_labels[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(img_paths_temp, img_labels_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.img_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, img_paths_temp, img_labels_temp):
        X = []
        for img in img_paths_temp:
            x = load_img(
            img,
            grayscale=False,
            color_mode="grayscale",
            target_size=(img_size,img_size),
            interpolation="nearest",
            )
            x = np_highpass_filter(x)
            #x = x + 10**(-10)
            #x = np.log(x)
            x = np.expand_dims(x, axis=0)
            X.append(x)

        X = np.vstack(X)
        #X = X[:, :, :, np.newaxis].astype('float32')
        #X_test = X_test[:, :, :, np.newaxis].astype('float32')
        return X,  np.array(img_labels_temp)#tf.keras.utils.to_categorical(img_labels_temp, num_classes=self.n_classes)

def np_highpass_filter(
        feature_map,
        d0=100,
        norm=None
        ):

    def distance(point1,point2):
        return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

    def gaussianHP(D0,imgShape):
        base = np.zeros(imgShape[:2])
        rows, cols = imgShape[:2]
        center = (rows/2,cols/2)
        for x in range(cols):
            for y in range(rows):
                base[y,x] = 1 - exp(((-distance((y,x),center)**2)/(2*(D0**2))))
        return base

    original = np.fft.fft2(feature_map)
    center = np.fft.fftshift(original)
    HighPassCenter = center * gaussianHP(d0,np.shape(feature_map))
    HighPass = np.fft.ifftshift(HighPassCenter)
    inverse_HighPass = np.fft.ifft2(HighPass)
    return np.abs(inverse_HighPass)

def tf_highpass_filter(
        feature_map,
        d0=100,
        norm=None,
        dtype = "complex64"
        ):

    def distance(point1,point2):
        return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

    def gaussianHP(D0,imgShape):
        base = np.zeros(imgShape[:2])
        rows, cols = imgShape[:2]
        center = (rows/2,cols/2)
        for x in range(cols):
            for y in range(rows):
                base[y,x] = 1 - exp(((-distance((y,x),center)**2)/(2*(D0**2))))
        return base

    #original = np.fft.fft2(feature_map)
    feature_map = tf.cast(feature_map, dtype)
    original = tf.signal.fft2d(
    feature_map, name=None
    )

    #center = np.fft.fftshift(original)
    center = tf.signal.fftshift(
    original, axes=None, name=None
    )

    HighPassCenter = center * gaussianHP(d0,np.shape(feature_map))
    #HighPass = np.fft.ifftshift(HighPassCenter)
    HighPass = tf.signal.ifftshift(
    HighPassCenter, axes=None, name=None
    )

    #inverse_HighPass = np.fft.ifft2(HighPass)
    inverse_HighPass = tf.signal.ifft2d(
    HighPass, name=None
    )

    return tf.math.abs(inverse_HighPass)

def load_data_highpass(img_size, batch_size, csv_train, csv_test):
    df_train = pd.read_csv(csv_train, sep=',')
    #df_train = df_train[:100000]
    df_test = pd.read_csv(csv_test, sep=',')
    #df_test = df_test[:13000]
    df_train['spoof'] = df_train['spoof'].astype(int)
    df_test['spoof'] = df_test['spoof'].astype(int)
    print(df_test)

    train_cls_n = []
    for i in range(df_train['spoof'].nunique()):
        train_cls_n.append(df_train[df_train.spoof == int(i)].shape[0])
    print('train_cls_n = ', train_cls_n, len(train_cls_n))

    test_cls_n = []
    for i in range(df_test['spoof'].nunique()):
        test_cls_n.append(df_test[df_test.spoof == int(i)].shape[0])
    print('test_cls_n = ', test_cls_n, len(test_cls_n))         

    train_generator=DataGenerator_highpass(df_train['file'], df_train['spoof'], img_size, batch_size)
    test_generator=DataGenerator_highpass(df_test['file'], df_test['spoof'], img_size, batch_size)

    return train_generator, test_generator

def load_data(img_size, batch_size, csv_train, csv_test):
    df_train = pd.read_csv(csv_train, sep=',')
    #df_train = df_train[:100000]
    df_test = pd.read_csv(csv_test, sep=',')
    #df_test = df_test[:13000]
    df_train['spoof'] = df_train['spoof'].astype(str)
    df_test['spoof'] = df_test['spoof'].astype(str)
    print(df_test)

    train_cls_n = []
    for i in range(df_train['spoof'].nunique()):
        train_cls_n.append(df_train[df_train.spoof == str(i)].shape[0])
    print('train_cls_n = ', train_cls_n, len(train_cls_n))

    test_cls_n = []
    for i in range(df_test['spoof'].nunique()):
        test_cls_n.append(df_test[df_test.spoof == str(i)].shape[0])
    print('test_cls_n = ', test_cls_n, len(test_cls_n))         

    train_datagen=ImageDataGenerator()

    train_generator=train_datagen.flow_from_dataframe(
    dataframe=df_train,
    color_mode="grayscale",
    x_col="file",
    y_col="spoof",
    batch_size=batch_size,
    shuffle=False,
    target_size=(img_size,img_size),
    class_mode='binary')

    test_datagen=ImageDataGenerator()

    test_generator=test_datagen.flow_from_dataframe(
    dataframe=df_test,
    color_mode="grayscale",
    x_col="file",
    y_col="spoof",
    batch_size=batch_size,
    shuffle=False,
    target_size=(img_size,img_size),
    class_mode='binary')
    return train_generator, test_generator

def fit(model, train_generator, test_generator, initial_epoch, final_epoch, dst_pth, num_workers, max_queue_size, initial_lr, factor, patience, min_lr):
    def scheduler(epoch, lr):
        return lr * 0.6
            
    optimizer = optimizers.RMSprop(lr=initial_lr)

    tensor_board = TensorBoard(log_dir='{}/tensorboard'.format(dst_pth), histogram_freq=0, write_graph=True,
                               write_images=True, update_freq=n_epochs, profile_batch=0)

    lr_scheduler = ReduceLROnPlateau(#monitor='val_accuracy',
                                    #mode='max',
                                    monitor='val_loss',
                                    mode='min',
                                    factor=factor,
                                    patience=patience,
                                    verbose=1,
                                    min_lr=min_lr
                                    )
    ckpt = ModelCheckpoint(
                "{}/c3ae-128-".format(dst_pth)+
                "epoch:{epoch:03d}-val_loss:{val_loss:.4f}-val_accuracy:{val_accuracy:.4f}.h5", 
                verbose=1,
                save_best_only=False, 
                period=1)

    callback = [lr_scheduler,
                tensor_board,
                ckpt
                ]

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x=train_generator,
                        callbacks=callback,
                        initial_epoch=initial_epoch,
                        epochs=final_epoch,
                        validation_data=test_generator,
                        workers=num_workers,
                        max_queue_size=max_queue_size,
                        verbose=1
                        )
    return model, history

def get_model(wght_pth):
    return tf.keras.models.load_model(wght_pth)

if __name__ == '__main__':
    n_epochs = 10
    initial_epoch = 0
    final_epoch = initial_epoch + n_epochs
    img_size = 128
    batch_size = 100
    num_workers = 8
    max_queue_size = 8
    factor = 0.9
    patience = 1
    initial_lr = 0.0017
    min_lr = 0.0000001
    dst_pth = '/home/yandex/igor/julia/arcface_more_classes'
    csv_train = '/mnt/data/lossless_train_04102022_crops.csv'
    csv_test = '/mnt/data/lossless_val_04102022_crops.csv'

    model = arcface_model(img_size)
    
    #wght_pth = '/home/yandex/igor/julia/arcface/strange/c3ae2-128-epoch:002-val_loss:0.0261-val_accuracy:1.0000.h5'
    #model = arcface_get_model(wght_pth)
    print(model.summary())

    train_generator, test_generator = arcface_load_data(img_size, batch_size, csv_train, csv_test)
    model, history = fit(model, train_generator, test_generator, initial_epoch, final_epoch, dst_pth, num_workers, max_queue_size, initial_lr, factor, patience, min_lr)
