from tensorflow.keras.models import Model

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Layer
from tensorflow.keras.preprocessing.image import  load_img

import tensorflow as tf

import pandas as pd
import numpy as np
import math

from train_arcface import *
 
def _resolve_training(layer, training):
    if training is None:
        training = tf.keras.backend.learning_phase()
    if isinstance(training, int):
        training = bool(training)
    if not layer.trainable:
        # When the layer is not trainable, override the value
        training = False
    return training
    
class ArcFace(Layer):
    """
    Implementation of ArcFace layer. Reference: https://arxiv.org/abs/1801.07698
    
    Arguments:
      num_classes: number of classes to classify
      s: scale factor
      m: margin
      regularizer: weights regularizer
    """
    def __init__(self,
                 n_classes,
                 s=30.0,
                 m=0.5,
                 regularizer=None,
                 name='arcface',
                 **kwargs):
        
        super().__init__(name=name, **kwargs)
        self._n_classes = n_classes
        self._s = float(s)
        self._m = float(m)
        self._regularizer = regularizer

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_classes': self._n_classes,
            's': self._s,
            'm': self._m,
        })
        return config

    def build(self, input_shape):
        embedding_shape, label_shape = input_shape
        self._w = self.add_weight(shape=(embedding_shape[-1], self._n_classes),
                                  initializer='glorot_uniform',
                                  trainable=True,
                                  regularizer=self._regularizer,
                                  name='cosine_weights')

    def call(self, inputs, training=None):
        """
        During training, requires 2 inputs: embedding (after backbone+pool+dense),
        and ground truth labels. The labels should be sparse (and use
        sparse_categorical_crossentropy as loss).
        """
        embedding, label = inputs

        # Squeezing is necessary for Keras. It expands the dimension to (n, 1)
        label = tf.reshape(label, [-1], name='label_shape_correction')

        # Normalize features and weights and compute dot product
        x = tf.nn.l2_normalize(embedding, axis=1, name='normalize_prelogits')
        w = tf.nn.l2_normalize(self._w, axis=0, name='normalize_weights')
        cosine_sim = tf.matmul(x, w, name='cosine_similarity')

        training = _resolve_training(self, training)
        if not training:
            # We don't have labels if we're not in training mode
            return self._s * cosine_sim
        else:
            one_hot_labels = tf.one_hot(label,
                                        depth=self._n_classes,
                                        name='one_hot_labels')
            theta = tf.math.acos(tf.keras.backend.clip(
                    cosine_sim, -1.0 + tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon()))
            selected_labels = tf.where(tf.greater(theta, math.pi - self._m),
                                       tf.zeros_like(one_hot_labels),
                                       one_hot_labels,
                                       name='selected_labels')
            final_theta = tf.where(tf.cast(selected_labels, dtype=tf.bool),
                                   theta + self._m,
                                   theta,
                                   name='final_theta')
            output = tf.math.cos(final_theta, name='cosine_sim_with_margin')
            return self._s * output


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, img_paths, img_labels, img_size, batch_size, n_classes=500, shuffle=False):
        'Initialization'
        self.batch_size = batch_size
        self.img_size = img_size
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

        return [X, y], y

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
            color_mode="rgb",
            target_size=(self.img_size,self.img_size),
            interpolation="nearest",
            )
            x = np.expand_dims(x, axis=0)
            X.append(x)
        X = np.vstack(X)
        #X = X[:, :, :, np.newaxis].astype('float32')
        #X_test = X_test[:, :, :, np.newaxis].astype('float32')
        return X, np.array(img_labels_temp)#.astype(np.uint8)#tf.keras.utils.to_categorical(img_labels_temp, num_classes=self.n_classes)

def arcface_model(img_size, n_classes, s, m):

    input_tensor = Input(shape=(img_size, img_size, 3))
    label = Input(shape=(1,))

    base_model_dct = build_shared_plain_network(input_tensor, mode='dct')
    
    x = Dense(500, kernel_initializer='he_normal')(base_model_dct.output)
    x = BatchNormalization()(x)
    output = ArcFace(n_classes=n_classes, s=s, m=m)([x, tf.cast(label, tf.uint8)])

    model = Model([input_tensor, label], output)
    return model

def arcface_load_data(img_size, batch_size, csv_train, csv_test):
    df_train = pd.read_csv(csv_train, sep=',')
    df_train = df_train[:100000]
    df_test = pd.read_csv(csv_test, sep=',')
    df_test = df_test[:130000]
    df_train['label'] = df_train['label'].astype(int)
    df_test['label'] = df_test['label'].astype(int)
    print(df_test)

    train_cls_n = []
    for i in range(df_train['label'].nunique()):
        train_cls_n.append(df_train[df_train.label == int(i)].shape[0])
    print('train_cls_n = ', train_cls_n, len(train_cls_n))

    test_cls_n = []
    for i in range(df_test['label'].nunique()):
        test_cls_n.append(df_test[df_test.label == int(i)].shape[0])
    print('test_cls_n = ', test_cls_n, len(test_cls_n))         

    train_generator=DataGenerator(df_train['file'], df_train['label'], img_size, batch_size)
    test_generator=DataGenerator(df_test['file'], df_test['label'], img_size, batch_size)

    return train_generator, test_generator

def finetune_arcface(wght_pth, n_classes, s, m):
    model_old = get_model(wght_pth)
    for layer in model_old.layers[:-1]:
        layer.trainable = False
    layer_name = 'dense_10'
    model2= Model(inputs=model_old.input, outputs=model_old.get_layer(layer_name).output)
    # Now add a new layer to the model
    input_tensor = model2.input
    label = Input(shape=(1,), name='label_input')
    output = ArcFace(n_classes=n_classes, s=s, m=m)([model2.output, tf.cast(label, tf.uint8)])
    model = Model([input_tensor, label], output)
    return model

def arcface_get_model(wght_pth):
    return tf.keras.models.load_model(wght_pth, custom_objects={'ArcFace': ArcFace})
