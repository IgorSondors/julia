import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D, Conv2D, Input, Flatten, concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, AveragePooling2D, multiply, Lambda
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler, ReduceLROnPlateau
import tensorflow as tf

import pandas as pd
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def WMWStatistic(gamma=0.4, p=2):
    """Computes a loss function based on an approximation of the normalized
    Wilcoxon-Mann-Whitney (WMW) statistic.
    The normalized WMW statistic can be shown to be equal the AUC-ROC. However,
    it is a step function so it is not differentiable. The normalized WCW
    statistic can be approximated with a smooth, differentiable function
    which makes the approximated version an ideal loss function for optimizing
    for the AUC-ROC metric.
    The loss function has two parameters, gamma and p, which are recommended
    to be kept between 0.1 to 0.7 and at 2 or 3, respectively.
    For more information:
    Optimizing Classifier Performance via an Approximation to the
    Wilcoxon-Mann-Whitney Statistic. Yan, Lian and Dodier, Robert H. and Mozer,
    Michael and Wolniewicz, Richard H. International Conference on Machine
    Learning (2003).
    """

    def loss(y_true, y_pred):
        # Convert labels and predictions to float64.
        y_true = tf.cast(y_true, dtype="float64")
        y_pred = tf.cast(y_pred, dtype="float64")

        # Boolean vector for determining positive and negative labels.
        boolean_vector = tf.greater_equal(y_true, 0.5)

        # Mask predictions to seperate true positive and negatives.
        positive_predictions = tf.boolean_mask(y_pred, boolean_vector)
        negative_predictions = tf.boolean_mask(y_pred, ~boolean_vector)

        # Obtain size of new arrays.
        m = tf.reduce_sum(tf.cast(boolean_vector, dtype="float64"))
        n = tf.reduce_sum(tf.cast(~boolean_vector, dtype="float64"))

        # Reshape arrays into original shape.
        positive_predictions = tf.reshape(positive_predictions, shape=(m, 1))
        negative_predictions = tf.reshape(negative_predictions, shape=(n, 1))

        # Convert gamma parameter to float64.
        gamma_array = tf.constant(gamma, dtype="float64")

        # Broadcast gamma parameter to MxN matrix.
        gamma_array = tf.broadcast_to(gamma_array, shape=(m, n))

        # Broadcast positive predictions to MxN.
        positive_predictions = tf.broadcast_to(positive_predictions, shape=(m, n))

        # Broadcast negative predictions to NxM then transpose.
        negative_predictions = tf.transpose(tf.broadcast_to(negative_predictions,
                                                            shape=(n, m)))

        # Subtract positive predictions matrix from negative predictions matrix.
        sub_neg_pos = tf.subtract(negative_predictions, positive_predictions)

        # Add gamma matrix to subtracted negative/positive matrix.
        add_gamma = tf.add(sub_neg_pos, gamma_array)

        # Check if positive predictions are less than negative predictions plus
        # gamma.
        inequality_check = tf.math.less(tf.subtract(positive_predictions,
                                                    negative_predictions), gamma)

        # Convert Boolean values to float64.
        inequality_check = tf.cast(inequality_check, dtype="float64")

        # Element-wise multiplication which effectively masks values that do not
        # meet inequality criterion.
        inequality_mask = tf.math.multiply(inequality_check, add_gamma)

        # Element-wise raise to power P.
        raise_to_p = tf.math.pow(inequality_mask, p)

        # Sum all elements.
        return tf.reduce_sum(raise_to_p) / tf.cast(tf.shape(y_true)[0], tf.float64)

    return loss

def get_create_gen_balanced(preprocess_function):
    def create_train_gen(path, batch_size=32, shuffle=True, augment=True):
        df = pd.read_csv(path, sep=',')
        df = df[:10000]
        df['spoof'] = df['spoof'].astype(str)

        train_cls_n = []
        for i in range(df['spoof'].nunique()):
            train_cls_n.append(df[df.spoof == str(i)].shape[0])
        print('train_cls_n = ', train_cls_n, len(train_cls_n))

        gen_pos = [tuple(row) for row in df[['file', 'spoof']].values if tuple(row)[1] == 1]
        gen_neg = [tuple(row) for row in df[['file', 'spoof']].values if tuple(row)[1] == 0]

        ds_pos = tf.data.Dataset.from_generator(lambda: gen_pos, (tf.string, tf.int32))
        ds_pos = ds_pos.shuffle(len(gen_pos)).repeat()

        ds_neg = tf.data.Dataset.from_generator(lambda: gen_neg, (tf.string, tf.int32))
        ds_neg = ds_neg.shuffle(len(gen_neg)).repeat()

        ds = tf.data.Dataset.zip((ds_pos, ds_neg))

        ds = ds.flat_map(lambda ex_pos, ex_neg: tf.data.Dataset.from_tensors(ex_pos).concatenate(tf.data.Dataset.from_tensors(ex_neg)))

        ds = ds.map(preprocess_function, num_parallel_calls=batch_size)
        ds = ds.batch(batch_size)

        ds.batch_size = batch_size
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        ds.len_filenames = 2 * min(len(gen_pos), len(gen_neg)) - 1
        print('ds = ', ds)
        return ds

    return create_train_gen


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

    input_tensor = Input(shape=(img_size, img_size, 3))

    base_model_dct = build_shared_plain_network(input_tensor, mode='dct')
    op_dct = Dropout(.5)(base_model_dct.output)
    x_dct = Dense(32, activation='swish')(op_dct)

    output_tensor = Dense(1, activation='sigmoid')(x_dct)
    model = Model(inputs=input_tensor, outputs=output_tensor)

    #model = tf.keras.models.experimental.SharpnessAwareMinimization(
    #model, rho=0.05, num_batch_splits=None, name=None
    #)
    
    return model#.build((None, img_size, img_size, 3))#model


def load_data(img_size, batch_size, csv_train, csv_test):
    
    train_generator=get_create_gen_balanced(ImageDataGenerator())

    test_generator=get_create_gen_balanced(ImageDataGenerator())
    #test_datagen=ImageDataGenerator()


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

    callback = [#LearningRateScheduler(scheduler),
                lr_scheduler,
                tensor_board#,
                #ckpt
                ]

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print('train_generator = ', train_generator)
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
    n_epochs = 2
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
    dst_pth = '/home/yandex/igor/julia/dct_sam'
    csv_train = '/mnt/data/lossless_train_04102022_crops.csv'
    csv_test = '/mnt/data/lossless_val_04102022_crops.csv'

    model = create_model(img_size)
    #wght_pth = 'path_to_the_checkpoint_you_want_to_start_from'
    #model = get_model(wght_pth)
    #print(model.summary())

    train_generator, test_generator = load_data(img_size, batch_size, csv_train, csv_test)
    model, history = fit(model, train_generator, test_generator, initial_epoch, final_epoch, dst_pth, num_workers, max_queue_size, initial_lr, factor, patience, min_lr)


    tf.keras.utils.plot_model(
    model,
    to_file="{}/dct_sam.png".format(dst_pth),
    show_shapes=True,
    show_dtype=False,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=96,
    )

    print(model.summary())