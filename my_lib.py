import numpy as np
import matplotlib.pyplot as plt
import json
import psutil
import glob
import nibabel as nib
import os
import skimage.io as io
import skimage.transform as trans
from sklearn.preprocessing import normalize
from scipy.ndimage import zoom
from keras.models import *
from keras.layers import *
import keras.backend as K
from keras.activations import *
from keras.optimizers import *
from keras.callbacks import Callback
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import keras.initializers as initializers
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.initializers import random_normal
from keras.regularizers import l1_l2, l2
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.regularizers import l2
from keras.activations import *
from keras.activations import *

def print_memory_use():
    '''
    Function which prints current python memory usage in megabytes
    '''
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss/1e9)

# What value maps to what class
mapping = {
    0: "Null class",
    1: "Necrotic and non-enhancing tumor core",
    2: "Edema",
    4: "GD-enhancing tumor"
}

mapping2 = {
    0: "Null class",
    1: "Tumor",
}

def create_train_test_split():
    '''
    Function which creates a train-, validation- and test split for 335 samples with the
    proportions 70%, 15%, 15%
    :return: A tuple containing 3 lists each with the indices of train, validation and test set respectively
    '''
    # There is 335 patients in total. -> indices [0, 334]
    n = 335
    np.random.seed(42)
    ind = np.arange(n)
    np.random.shuffle(ind)
    ind1 = int(np.ceil(len(ind) * 0.7))
    ind2 = int(np.ceil(len(ind) * 0.85))
    ind3 = len(ind)

    train_ind = ind[0:ind1]
    val_ind = ind[ind1:ind2]
    test_ind = ind[ind2:ind3]
    return train_ind, val_ind, test_ind

def OHE(Y):
    '''
    :param Y: A slice containing original BraTS-data with classes {0,1,2,4}
    :return: A slice where classes 1 and 4 has been merged and this has been one hot encoded
    '''
    shape = Y.shape
    one_hot_enc = np.zeros(list(shape) + [2])
    temp = np.zeros(shape)
    temp2 = np.ones(shape)

    ind1 = Y == 1
    ind2 = Y == 4
    temp[ind1] = 1
    temp[ind2] = 1

    temp2 = temp2 - temp

    one_hot_enc[:, :, 0] = temp
    one_hot_enc[:, :, 1] = temp2
    return one_hot_enc

def OHE1(Y, mapping):
    '''
    Takes in a picture as a matrix with labels and returns a one hot encoded tensor

    Parameters:
    Y is the picture
    Mapping is what value corresponds to what label

    Returns:
    A tensor with a channel for each label.
    '''
    shape = Y.shape
    labels = mapping.keys()
    one_hot_enc = np.zeros(list(shape) + [len(labels)])

    for i, label in enumerate(labels):
        temp = np.zeros(shape)
        ind = Y == label
        temp[ind] = 1
        one_hot_enc[:, :, i] = temp
    return one_hot_enc

def OHE_uncoding(y, mapping):
    result = np.argmax(y, axis=-1)
    labels = mapping.keys()
    temp = np.zeros(result.shape)
    for i, label in enumerate(labels):
        ind = result == i
        temp[ind] = label
    return temp

def load_patients_numpy(path_to_folder, indices, cropping=False):
    '''
    :param path_to_folder: The path to the folder which contain the patients saved one by one in .npz format
    :param indices: A list with the indices, range [0, 335], which one wants to load to memory
    :return: returns one numpy array with training data and one numpy array with the corresponding one hot
    encoded labels.
    '''
    start = True
    for count, i in enumerate(indices):
        if count % 10 == 0:
            print("Patient: ", count)
        y_up = 40
        y_down = 216
        x_left = 40
        x_right = 216
        H = y_down - y_up
        W = x_right - x_left

        if start:
            data = np.load(path_to_folder + '/patient-' + str(i) + '.npz')
            X = data['arr_0'][0]
            Y = data['arr_0'][1]

            if cropping:
                X = X[:, x_left:x_right, y_up:y_down, :]
                Y = Y[:, x_left:x_right, y_up:y_down, :]

            start = False
        else:
            data = np.load(path_to_folder + '/patient-' + str(i) + '.npz')
            temp_X = data['arr_0'][0]
            temp_Y = data['arr_0'][1]

            if cropping:
                temp_X = temp_X[:, x_left:x_right, y_up:y_down, :]
                temp_Y = temp_Y[:, x_left:x_right, y_up:y_down, :]

            X = np.concatenate((X, temp_X), axis=0)
            Y = np.concatenate((Y, temp_Y), axis=0)

    return X, Y

def load_patients(i, j, base_path="", rescale=None):
    '''
    Function which loads patients from BraTS data
    :param i: First patient to be loaded
    :param j: From patient i to j load all patients
    :param base_path: Specifies where data is
    :return: A tuple with data in the first place and labels in the second place.
    Data has shape (n,240,240,1) where n is the number of slices from patient i to j who contains tumors
    and the labels has has shape (n, 240, 240, 2) which is a pixelwise binary softmax.
    '''
    assert j >= i, 'j>i has to be true, you have given an invalid range of patients.'
    path = base_path + "MICCAI_BraTS_2019_Data_Training/*/*/*"
    wild_t1ce = path + "_t1ce.nii.gz"
    wild_gt = path + "_seg.nii.gz"

    t1ce_paths = glob.glob(wild_t1ce)
    gt_paths = glob.glob(wild_gt)

    num_patients = j - i
    ind = []
    #fixme: the list and the set patients should be made into a dictionary
    patients = set({})
    num_non_empty_slices = 0
    labels_of_interest = set([1, 4])

    for k in range(i, j):
        path_gt = gt_paths[k]
        img_gt = nib.load(path_gt)
        img_gt = img_gt.get_fdata()
        curr_patient = []
        for l in range(img_gt.shape[-1]):
            labels_in_slice = set(np.unique(img_gt[:, :, l]))
            if labels_of_interest.issubset(labels_in_slice):
                curr_patient.append(l)
                num_non_empty_slices += 1
                patients.add(k)
        if len(curr_patient) > 0:
            ind.append(curr_patient)

    image_data = np.zeros((1, 240, 240, num_non_empty_slices))
    labels = np.zeros((num_non_empty_slices, 240, 240))
    OHE_labels = np.zeros((num_non_empty_slices, 240, 240, 2))
    next_ind = 0

    for k, y in enumerate(patients):
        print('Patient: ' + str(y))
        curr_ind = ind[k]

        path_t1ce = t1ce_paths[y]
        path_gt = gt_paths[y]

        img_t1ce = nib.load(path_t1ce)
        img_gt = nib.load(path_gt)

        img_t1ce = img_t1ce.get_fdata()
        img_gt = img_gt.get_fdata()

        # This code is necessary when we will use the data from Asgeir
        if rescale:
            img_gt = zoom(img_gt, rescale, order=0)
            img_t1ce = zoom(img_t1ce, rescale, order=0)

        temp = 0
        for l, x in enumerate(curr_ind):
            image_data[0, :, :, next_ind + l] = img_t1ce[:, :, x]
            labels[next_ind + l, :, :] = img_gt[:, :, x]
            temp += 1
        next_ind += temp

    for l in range(num_non_empty_slices):
        image_data[0, :, :, l] = normalize(image_data[0, :, :, l])
        OHE_labels[l, :, :, :] = OHE(labels[l, :, :])

    # The last axis will become the first axis
    image_data = np.moveaxis(image_data, -1, 0)
    image_data = np.moveaxis(image_data, 1, 3)
    return (image_data, OHE_labels, patients)

def convert_brats_to_asgeir(Y):
    '''
    Convert the BraTS-data to binary classification task of the tumor core
    :param Y: The labels which are to be converted
    :return: Returns the converted labels
    '''
    tumor_core = Y[:, :, :, 1] + Y[:, :, :, 3]
    tumor_core = np.expand_dims(tumor_core, axis=3)
    new_background = Y[:, :, :, 0] + Y[:, :, :, 2]
    new_background = np.expand_dims(new_background, axis=3)
    Y_converted = np.concatenate((new_background, tumor_core), axis=3)
    return Y_converted

def conv_block(input_, num_kernels, kernel_size, act_func, drop_rate):
    conv = Conv2D(num_kernels, kernel_size, activation=act_func, padding='same', kernel_initializer='he_normal')(input_)
    conv = Conv2D(num_kernels, kernel_size, activation=act_func, padding='same', kernel_initializer='he_normal')(conv)
    drop = Dropout(drop_rate)(conv)
    return conv

def conv_block_resnet(input_, num_kernels, kernel_size, act_func, drop_rate, input_size):
    argz = [num_kernels, kernel_size]
    kwargz = {
        'activation':act_func, 
        'padding':'same', 
        'kernel_initializer':'he_normal', 
        'kernel_regularizer':l1_l2(l1=0.01, l2=0.01)}
    
    conv = Conv2D(*argz, **kwargz)(input_)
    conv = Conv2D(*argz, **kwargz)(conv)
    conv = Conv2D(
        input_size[-1], 
        (1,1), 
        activation = 'linear', 
        padding = 'same', 
        kernel_initializer = 'he_normal')(conv)
    conv = Dropout(drop_rate)(conv)
    merge = Add()([input_, conv])
    # Since we do not use BatchNormalization we shouldn't use BatchNorm
    #merge = BatchNormalization()(merge)
    merge = Activation(act_func)(merge)
    return merge

def down_sampling_block(input_, act_func, num_kernels, drop_rate, input_size, res=False):
    if res:
        skip = conv_block_resnet(input_=input_, num_kernels=num_kernels, kernel_size=(3,3),
                                 act_func=act_func, drop_rate=drop_rate, input_size=input_size)
    else:
        skip = conv_block(input_, num_kernels=num_kernels, kernel_size=(3,3), act_func=act_func, drop_rate=drop_rate)
    pool = MaxPooling2D(pool_size = (2, 2))(skip)
    return skip, pool

def up_sampling_block(input_, skip, act_func, num_kernels, drop_rate, input_size, res=False):
    up = UpSampling2D(size = (2, 2))(input_)
    merge = concatenate([skip, up], axis = 3)
    if res:
        conv = conv_block_resnet(up, num_kernels=num_kernels, kernel_size=(3,3),
                                 act_func=act_func, drop_rate=drop_rate, input_size=input_size)
    else:
        conv = conv_block(merge, num_kernels, (3,3), act_func, drop_rate)
    return conv

def unet_clean(pretrained_weights = None, input_size = (256, 256, 1), num_classes=2, learning_rate=1e-4, act_func='relu', res=False, metrics=None):
    # Encoder
    inputs = Input(input_size)
    skip1, pool1 = down_sampling_block(inputs, act_func, num_kernels=64, drop_rate=0, input_size=input_size, res=res)
    skip2, pool2 = down_sampling_block(pool1, act_func, num_kernels=128, drop_rate=0, input_size=input_size, res=res)
    skip3, pool3 = down_sampling_block(pool2, act_func, num_kernels=256, drop_rate=0, input_size=input_size, res=res)
    skip4, pool4 = down_sampling_block(pool3, act_func, num_kernels=512, drop_rate=0, input_size=input_size, res=res)

    #Bottleneck
    conv5 = conv_block(pool4, 1024, 3, act_func, drop_rate=0)

    # Decoder
    conv6 = up_sampling_block(conv5, skip4, act_func, 512, drop_rate = 0, input_size = input_size, res=res)
    conv7 = up_sampling_block(conv6, skip3, act_func, 256, drop_rate = 0, input_size = input_size, res=res)
    conv8 = up_sampling_block(conv7, skip2, act_func, 128, drop_rate = 0, input_size = input_size, res=res)
    conv9 = up_sampling_block(conv8, skip1, act_func, 64, drop_rate = 0, input_size = input_size, res=res)
    conv9 = Conv2D(num_classes, 1, activation=act_func, padding='same', kernel_initializer='he_normal')(conv9)

    reshape = Reshape((num_classes, input_size[0] * input_size[1]), input_shape = (num_classes, input_size[0], input_size[1]))(conv9)
    permute = Permute((2, 1))(reshape)
    activation = Softmax(axis=-1, dtype='float64')(permute)

    model = Model(input=inputs, output=activation)
    model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=metrics)
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    return model

def unet_depth(pretrained_weights = None, input_size = (256, 256, 1), num_classes=2, learning_rate=1e-4,
               act_func='relu', res=False, depth=4, num_kernels=[64, 128, 256, 512], metrics=None):
    assert depth == len(num_kernels), 'Depth and number of kernel sizes must be equal'

    encoder = []
    inputs = Input(input_size)
    for i in range(depth):
        if i == 0:
            skip, conv = down_sampling_block(inputs, act_func, num_kernels=num_kernels[i], drop_rate=0, input_size = input_size, res=res)
            result = [skip, conv]
            encoder.append(result)
        else:
            skip, conv = down_sampling_block(encoder[i-1][1], act_func, num_kernels=num_kernels[i], drop_rate=0, input_size=input_size, res=res)
            result = [skip, conv]
            encoder.append(result)

    bottleneck = conv_block(encoder[depth - 1][1], 1024, 3, act_func, drop_rate=0.2)

    decoder = []
    for i in range(depth):
        if i == 0:
            skip = encoder[depth - 1][0]
            decoder.append(up_sampling_block(bottleneck, skip, act_func, num_kernels=num_kernels[depth - i - 1], drop_rate=0, input_size=input_size, res=res))
        else:
            skip = encoder[depth - i - 1][0]
            decoder.append(up_sampling_block(decoder[i - 1], skip, act_func, num_kernels=num_kernels[depth - i - 1], drop_rate=0, input_size=input_size, res=res))

    # prepare for softmax
    conv = Conv2D(num_classes, 1, activation=act_func, padding='same', kernel_initializer='he_normal')(decoder[depth - 1])
    reshape = Reshape((num_classes, input_size[0] * input_size[1]), input_shape = (num_classes, input_size[0], input_size[1]))(conv)
    permute = Permute((2, 1))(reshape)
    activation = Softmax(axis=-1)(permute)

    # Compile model and load pretrained weights
    model = Model(input=inputs, output=activation)
    model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=metrics)
    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model

def unet_dong_et_al(input_size, num_classes, lr, metrics, drop_rate, loss, pretrained_weights=None):
    kernel_size = 3
    drop_rate = drop_rate
    conv_kwargs = {
        'strides': (1, 1),
        'padding': 'same',
        'activation': 'relu',
        'kernel_initializer': random_normal(stddev=0.01)
        #'activity_regularizer': l1_l2()
    }
    conv_transpose_kwargs = {
        'strides': (2, 2),
        'kernel_initializer': random_normal(stddev=0.01)
        #'activity_regularizer': l1_l2()
    }
    conv_kwargs_fin = {
        'strides': (1, 1),
        'padding': 'same',
        'activation': 'relu',
        'kernel_initializer': random_normal(stddev=0.01)
    }
    pooling_kwargs = {
        'pool_size': (2, 2),
        'padding': 'valid'
    }

    # Encoder
    inputs = Input(input_size)
    conv1 = Conv2D(64, kernel_size, **conv_kwargs)(inputs)
    conv1 = Dropout(rate=drop_rate)(conv1)
    conv1 = Conv2D(64, kernel_size, **conv_kwargs)(conv1)
    conv1 = Dropout(rate=drop_rate)(conv1)
    pool1 = MaxPooling2D(**pooling_kwargs)(conv1)

    conv2 = Conv2D(128, kernel_size, **conv_kwargs)(pool1)
    conv2 = Dropout(rate=drop_rate, )(conv2)
    conv2 = Conv2D(128, kernel_size, **conv_kwargs)(conv2)
    conv2 = Dropout(rate=drop_rate)(conv2)
    pool2 = MaxPooling2D(**pooling_kwargs)(conv2)

    conv3 = Conv2D(256, kernel_size, **conv_kwargs)(pool2)
    conv3 = Dropout(rate=drop_rate)(conv3)
    conv3 = Conv2D(256, kernel_size, **conv_kwargs)(conv3)
    conv3 = Dropout(rate=drop_rate)(conv3)
    pool3 = MaxPooling2D(**pooling_kwargs)(conv3)

    conv4 = Conv2D(512, kernel_size, **conv_kwargs)(pool3)
    conv4 = Dropout(rate=drop_rate)(conv4)
    conv4 = Conv2D(512, kernel_size, **conv_kwargs)(conv4)
    conv4 = Dropout(rate=drop_rate)(conv4)
    pool4 = MaxPooling2D(**pooling_kwargs)(conv4)

    # Bottleneck
    conv5 = Conv2D(1024, kernel_size, **conv_kwargs)(pool4)
    conv5 = Dropout(rate=drop_rate)(conv5)
    conv5 = Conv2D(1024, kernel_size, **conv_kwargs)(conv5)
    conv5 = Dropout(rate=drop_rate)(conv5)

    # Decoder
    up6 = Conv2DTranspose(512, (2, 2), **conv_transpose_kwargs)(conv5)
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(512, kernel_size, **conv_kwargs)(merge6)
    conv6 = Dropout(rate=drop_rate)(conv6)
    conv6 = Conv2D(512, kernel_size, **conv_kwargs)(conv6)
    conv6 = Dropout(rate=drop_rate)(conv6)

    up7 = Conv2DTranspose(256, (2, 2), **conv_transpose_kwargs)(conv6)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, kernel_size, **conv_kwargs)(merge7)
    conv7 = Dropout(rate=drop_rate)(conv7)
    conv7 = Conv2D(256, kernel_size, **conv_kwargs)(conv7)
    conv7 = Dropout(rate=drop_rate)(conv7)

    up8 = Conv2DTranspose(128, (2, 2), **conv_transpose_kwargs)(conv7)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, kernel_size, **conv_kwargs)(merge8)
    conv8 = Dropout(rate=drop_rate)(conv8)
    conv8 = Conv2D(128, kernel_size, **conv_kwargs)(conv8)
    conv8 = Dropout(rate=drop_rate)(conv8)

    up9 = Conv2DTranspose(64, (2, 2), **conv_transpose_kwargs)(conv8)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, kernel_size, **conv_kwargs)(merge9)
    conv9 = Dropout(rate=drop_rate)(conv9)
    conv9 = Conv2D(64, kernel_size, **conv_kwargs_fin)(conv9)
    conv9 = Dropout(rate=drop_rate)(conv9)

    # Correct dimensions
    conv9 = Conv2D(num_classes, 1, **conv_kwargs_fin)(conv9)
    activation = Softmax()(conv9)
    unet = Model(inputs=[inputs], outputs=[activation])

    unet.compile(optimizer=Adam(lr=lr), loss=loss, metrics=metrics)
    if (pretrained_weights):
        unet.load_weights(pretrained_weights)
    return unet

def single_stream_unet(input_size, num_classes, lr, metrics, loss, pretrained_weights=None):
    '''
    Function to build U-Net
    :param input_size: A tuple that specifies the input shape to the network
    :param num_classes: How many classes the classification problem at hand has
    :param lr: learning rate to use during optimization
    :param metrics: a list of the metrics to use
    :param loss: specify loss function either with a string which keras supports or with a function
    :param pretrained_weights: If there exists pretrained weights these can optionally be passed
    :return: Returns a compiled U-Net
    '''
    kernel_size = 3
    conv_kwargs = {
        'strides': (1, 1),
        'padding': 'same',
        'activation': 'relu',
        'kernel_initializer': 'he_normal',
        'kernel_regularizer': l2(0.001)
    }
    conv_transpose_kwargs = {
        'strides': (2, 2),
        'kernel_initializer': 'he_normal',
        'kernel_regularizer': l2(0.001)
    }
    conv_kwargs_fin = {
        'strides': (1, 1),
        'padding': 'same',
        'activation': 'relu',
        'kernel_initializer': 'he_normal',
        'kernel_regularizer': l2(0.001)
    }
    pooling_kwargs = {
        'pool_size': (2, 2),
        'padding': 'valid'
    }

    # Encoder
    inputs = Input(input_size)
    conv1 = Conv2D(64, kernel_size, **conv_kwargs)(inputs)
    conv1 = Conv2D(64, kernel_size, **conv_kwargs)(conv1)
    pool1 = MaxPooling2D(**pooling_kwargs)(conv1)

    conv2 = Conv2D(128, kernel_size, **conv_kwargs)(pool1)
    conv2 = Conv2D(128, kernel_size, **conv_kwargs)(conv2)
    pool2 = MaxPooling2D(**pooling_kwargs)(conv2)

    conv3 = Conv2D(256, kernel_size, **conv_kwargs)(pool2)
    conv3 = Conv2D(256, kernel_size, **conv_kwargs)(conv3)
    pool3 = MaxPooling2D(**pooling_kwargs)(conv3)

    conv4 = Conv2D(512, kernel_size, **conv_kwargs)(pool3)
    conv4 = Dropout(rate=0.1)(conv4)
    conv4 = Conv2D(512, kernel_size, **conv_kwargs)(conv4)
    conv4 = Dropout(rate=0.1)(conv4)
    pool4 = MaxPooling2D(**pooling_kwargs)(conv4)

    # Bottleneck
    conv5 = Conv2D(1024, kernel_size, **conv_kwargs)(pool4)
    conv5 = Dropout(rate=0.1)(conv5)
    conv5 = Conv2D(1024, kernel_size, **conv_kwargs)(conv5)
    conv5 = Dropout(rate=0.1)(conv5)

    # Decoder
    up6 = Conv2DTranspose(256, (2, 2), **conv_transpose_kwargs)(conv5)
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(512, kernel_size, **conv_kwargs)(merge6)
    conv6 = Conv2D(512, kernel_size, **conv_kwargs)(conv6)

    up7 = Conv2DTranspose(128, (2, 2), **conv_transpose_kwargs)(conv6)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, kernel_size, **conv_kwargs)(merge7)
    conv7 = Conv2D(256, kernel_size, **conv_kwargs)(conv7)

    up8 = Conv2DTranspose(64, (2, 2), **conv_transpose_kwargs)(conv7)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, kernel_size, **conv_kwargs)(merge8)
    conv8 = Conv2D(128, kernel_size, **conv_kwargs)(conv8)

    up9 = Conv2DTranspose(32, (2, 2), **conv_transpose_kwargs)(conv8)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, kernel_size, **conv_kwargs)(merge9)
    conv9 = Conv2D(64, kernel_size, **conv_kwargs_fin)(conv9)

    # Correct dimensions
    conv9 = Conv2D(num_classes, 1, **conv_kwargs_fin)(conv9)
    activation = Softmax()(conv9)

    unet = Model(inputs=[inputs], outputs=[activation])
    unet.compile(optimizer=Adam(lr=lr), loss=loss, metrics=metrics)
    if (pretrained_weights):
        unet.load_weights(pretrained_weights)
    return unet

def unet_dong_single_mod(input_size, num_classes, lr, metrics, loss, pretrained_weights=None):
    '''
    Function to build U-Net
    :param input_size: A tuple that specifies the input shape to the network
    :param num_classes: How many classes the classification problem at hand has
    :param lr: learning rate to use during optimization
    :param metrics: a list of the metrics to use
    :param loss: specify loss function either with a string which keras supports or with a function
    :param pretrained_weights: If there exists pretrained weights these can optionally be passed
    :return: Returns a compiled U-Net
    '''
    kernel_size = 3
    conv_kwargs = {
        'strides': (1, 1),
        'padding': 'same',
        'activation': 'relu',
        'kernel_initializer': 'he_normal',
        'kernel_regularizer': l2(0.001)
    }
    conv_transpose_kwargs = {
        'strides': (2, 2),
        'kernel_initializer': 'he_normal',
        'kernel_regularizer': l2(0.001)
    }
    conv_kwargs_fin = {
        'strides': (1, 1),
        'padding': 'same',
        'activation': 'relu',
        'kernel_initializer': 'he_normal',
        'kernel_regularizer': l2(0.001)
    }
    pooling_kwargs = {
        'pool_size': (2, 2),
        'padding': 'valid'
    }

    # Encoder
    inputs = Input(input_size)
    conv1 = Conv2D(32, kernel_size, **conv_kwargs)(inputs)
    conv1 = Conv2D(32, kernel_size, **conv_kwargs)(conv1)
    pool1 = MaxPooling2D(**pooling_kwargs)(conv1)

    conv2 = Conv2D(64, kernel_size, **conv_kwargs)(pool1)
    conv2 = Conv2D(64, kernel_size, **conv_kwargs)(conv2)
    pool2 = MaxPooling2D(**pooling_kwargs)(conv2)

    conv3 = Conv2D(64, kernel_size, **conv_kwargs)(pool2)
    conv3 = Conv2D(64, kernel_size, **conv_kwargs)(conv3)
    pool3 = MaxPooling2D(**pooling_kwargs)(conv3)

    conv4 = Conv2D(256, kernel_size, **conv_kwargs)(pool3)
    conv4 = Dropout(rate=0.1)(conv4)
    conv4 = Conv2D(256, kernel_size, **conv_kwargs)(conv4)
    conv4 = Dropout(rate=0.1)(conv4)
    pool4 = MaxPooling2D(**pooling_kwargs)(conv4)

    # Bottleneck
    conv5 = Conv2D(512, kernel_size, **conv_kwargs)(pool4)
    conv5 = Dropout(rate=0.1)(conv5)
    conv5 = Conv2D(512, kernel_size, **conv_kwargs)(conv5)
    conv5 = Dropout(rate=0.1)(conv5)

    # Decoder
    up6 = Conv2DTranspose(256, (2, 2), **conv_transpose_kwargs)(conv5)
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(256, kernel_size, **conv_kwargs)(merge6)
    conv6 = Conv2D(512, kernel_size, **conv_kwargs)(conv6)

    up7 = Conv2DTranspose(128, (2, 2), **conv_transpose_kwargs)(conv6)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(128, kernel_size, **conv_kwargs)(merge7)
    conv7 = Conv2D(128, kernel_size, **conv_kwargs)(conv7)

    up8 = Conv2DTranspose(64, (2, 2), **conv_transpose_kwargs)(conv7)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(64, kernel_size, **conv_kwargs)(merge8)
    conv8 = Conv2D(64, kernel_size, **conv_kwargs)(conv8)

    up9 = Conv2DTranspose(32, (2, 2), **conv_transpose_kwargs)(conv8)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(32, kernel_size, **conv_kwargs)(merge9)
    conv9 = Conv2D(32, kernel_size, **conv_kwargs_fin)(conv9)

    # Correct dimensions
    conv9 = Conv2D(num_classes, 1, **conv_kwargs_fin)(conv9)
    activation = Softmax()(conv9)

    unet = Model(inputs=[inputs], outputs=[activation])
    unet.compile(optimizer=Adam(lr=lr), loss=loss, metrics=metrics)
    if (pretrained_weights):
        unet.load_weights(pretrained_weights)
    return unet

def unet_dong_small(input_size, num_classes, lr, metrics, loss, pretrained_weights=None):
    '''
    Function to build U-Net
    :param input_size: A tuple that specifies the input shape to the network
    :param num_classes: How many classes the classification problem at hand has
    :param lr: learning rate to use during optimization
    :param metrics: a list of the metrics to use
    :param loss: specify loss function either with a string which keras supports or with a function
    :param pretrained_weights: If there exists pretrained weights these can optionally be passed
    :return: Returns a compiled U-Net
    '''
    kernel_size = 3
    conv_kwargs = {
        'strides': (1, 1),
        'padding': 'same',
        'activation': 'relu',
        'kernel_initializer': 'he_normal',
        'kernel_regularizer': l2(0.001)
    }
    conv_transpose_kwargs = {
        'strides': (2, 2),
        'kernel_initializer': 'he_normal',
        'kernel_regularizer': l2(0.001)
    }
    conv_kwargs_fin = {
        'strides': (1, 1),
        'padding': 'same',
        'activation': 'relu',
        'kernel_initializer': 'he_normal',
        'kernel_regularizer': l2(0.001)
    }
    pooling_kwargs = {
        'pool_size': (2, 2),
        'padding': 'valid'
    }

    # Encoder
    inputs = Input(input_size)
    conv1 = Conv2D(64, kernel_size, **conv_kwargs)(inputs)
    pool1 = MaxPooling2D(**pooling_kwargs)(conv1)

    conv2 = Conv2D(128, kernel_size, **conv_kwargs)(pool1)
    pool2 = MaxPooling2D(**pooling_kwargs)(conv2)

    conv3 = Conv2D(256, kernel_size, **conv_kwargs)(pool2)
    pool3 = MaxPooling2D(**pooling_kwargs)(conv3)

    conv4 = Conv2D(512, kernel_size, **conv_kwargs)(pool3)
    conv4 = Dropout(rate=0.1)(conv4)
    pool4 = MaxPooling2D(**pooling_kwargs)(conv4)

    # Bottleneck
    conv5 = Conv2D(1024, kernel_size, **conv_kwargs)(pool4)
    conv5 = Dropout(rate=0.1)(conv5)

    # Decoder
    up6 = Conv2DTranspose(256, (2, 2), **conv_transpose_kwargs)(conv5)
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(512, kernel_size, **conv_kwargs)(merge6)

    up7 = Conv2DTranspose(128, (2, 2), **conv_transpose_kwargs)(conv6)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, kernel_size, **conv_kwargs)(merge7)

    up8 = Conv2DTranspose(64, (2, 2), **conv_transpose_kwargs)(conv7)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, kernel_size, **conv_kwargs)(merge8)

    up9 = Conv2DTranspose(32, (2, 2), **conv_transpose_kwargs)(conv8)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, kernel_size, **conv_kwargs)(merge9)

    # Correct dimensions
    conv9 = Conv2D(num_classes, 1, **conv_kwargs_fin)(conv9)
    activation = Softmax()(conv9)

    unet = Model(inputs=[inputs], outputs=[activation])
    unet.compile(optimizer=Adam(lr=lr), loss=loss, metrics=metrics)
    if (pretrained_weights):
        unet.load_weights(pretrained_weights)
    return unet

def zscore_norm(MRI):
    '''
    :param MRI: Image data should be given patientwise
    :return: Normalized data
    '''
    MRI_f = MRI.reshape(-1)
    mask_MRI = MRI_f != 0
    MRI_f[mask_MRI] = (MRI_f[mask_MRI] - MRI_f[mask_MRI].mean())/MRI_f[mask_MRI].std()
    return MRI_f.reshape(MRI.shape)

def unet_res(input_size, num_classes, lr, metrics, loss, pretrained_weights=None):
    '''
    Function to build U-Net
    :param input_size: A tuple that specifies the input shape to the network
    :param num_classes: How many classes the classification problem at hand has
    :param lr: learning rate to use during optimization
    :param metrics: a list of the metrics to use
    :param loss: specify loss function either with a string which keras supports or with a function
    :param pretrained_weights: If there exists pretrained weights these can optionally be passed
    :return: Returns a compiled U-Net
    '''
    kernel_size = 3
    conv_kwargs = {
        'strides': (1, 1),
        'padding': 'same',
        'activation': 'relu',
        'kernel_initializer': 'he_normal',
        'kernel_regularizer': l2(0.001)
    }
    conv_skip_kwargs = {
        'strides': (1, 1),
        'padding': 'same',
        'activation': 'linear',
        'kernel_initializer': 'he_normal',
        'kernel_regularizer': l2(0.001)
    }
    conv_transpose_kwargs = {
        'strides': (2, 2),
        'kernel_initializer': 'he_normal',
        'kernel_regularizer': l2(0.001)
    }
    conv_kwargs_fin = {
        'strides': (1, 1),
        'padding': 'same',
        'activation': 'relu',
        'kernel_initializer': 'he_normal',
        'kernel_regularizer': l2(0.001)
    }
    pooling_kwargs = {
        'pool_size': (2, 2),
        'padding': 'valid'
    }

    # Encoder
    inputs = Input(input_size)
    skipconv1 = Conv2D(64, 1, **conv_skip_kwargs)(inputs)
    conv1 = Conv2D(64, kernel_size, **conv_kwargs)(inputs)
    conv1 = Conv2D(64, kernel_size, **conv_skip_kwargs)(conv1)
    merge1 = Add()([skipconv1, conv1])
    merge1 = Activation('relu')(merge1)
    pool1 = MaxPooling2D(**pooling_kwargs)(merge1)

    skipconv2 = Conv2D(128, 1, **conv_skip_kwargs)(pool1)
    conv2 = Conv2D(128, kernel_size, **conv_kwargs)(pool1)
    conv2 = Conv2D(128, kernel_size, **conv_skip_kwargs)(conv2)
    merge2 = Add()([skipconv2, conv2])
    merge2 = Activation('relu')(merge2)
    pool2 = MaxPooling2D(**pooling_kwargs)(merge2)

    skipconv3 = Conv2D(256, 1, **conv_skip_kwargs)(pool2)
    conv3 = Conv2D(256, kernel_size, **conv_kwargs)(pool2)
    conv3 = Conv2D(256, kernel_size, **conv_skip_kwargs)(conv3)
    merge3 = Add()([skipconv3, conv3])
    merge3 = Activation('relu')(merge3)
    pool3 = MaxPooling2D(**pooling_kwargs)(merge3)

    skipconv4 = Conv2D(512, 1, **conv_skip_kwargs)(pool3)
    conv4 = Conv2D(512, kernel_size, **conv_kwargs)(pool3)
    conv4 = Dropout(rate=0.1)(conv4)
    conv4 = Conv2D(512, kernel_size, **conv_skip_kwargs)(conv4)
    conv4 = Dropout(rate=0.1)(conv4)
    merge4 = Add()([skipconv4, conv4])
    merge4 = Activation('relu')(merge4)
    pool4 = MaxPooling2D(**pooling_kwargs)(merge4)

    # Bottleneck
    skipconv5 = Conv2D(1024, 1, **conv_skip_kwargs)(pool4)
    conv5 = Conv2D(1024, kernel_size, **conv_kwargs)(pool4)
    conv5 = Dropout(rate=0.1)(conv5)
    conv5 = Conv2D(1024, kernel_size, **conv_kwargs)(conv5)
    conv5 = Dropout(rate=0.1)(conv5)
    merge5 = Add()([skipconv5, conv5])
    merge5 = Activation('relu')(merge5)
    pool5 = MaxPooling2D(**pooling_kwargs)(merge5)

    # Decoder
    up6 = Conv2DTranspose(256, (2, 2), **conv_transpose_kwargs)(conv5)
    merge6 = concatenate([conv4, up6], axis=3)
    skipconv6 = Conv2D(512, 1, **conv_skip_kwargs)(merge6)
    conv6 = Conv2D(512, kernel_size, **conv_kwargs)(merge6)
    conv6 = Conv2D(512, kernel_size, **conv_kwargs)(conv6)
    conv6 = Add()([skipconv6, conv6])
    conv6 = Activation('relu')(conv6)

    up7 = Conv2DTranspose(256, (2, 2), **conv_transpose_kwargs)(conv6)
    merge7 = concatenate([conv3, up7], axis=3)
    skipconv7 = Conv2D(512, 1, **conv_skip_kwargs)(merge7)
    conv7 = Conv2D(512, kernel_size, **conv_kwargs)(merge7)
    conv7 = Conv2D(512, kernel_size, **conv_kwargs)(conv7)
    conv7 = Add()([skipconv7, conv7])
    conv7 = Activation('relu')(conv7)

    up8 = Conv2DTranspose(256, (2, 2), **conv_transpose_kwargs)(conv7)
    merge8 = concatenate([conv2, up8], axis=3)
    skipconv8 = Conv2D(512, 1, **conv_skip_kwargs)(merge8)
    conv8 = Conv2D(512, kernel_size, **conv_kwargs)(merge8)
    conv8 = Conv2D(512, kernel_size, **conv_kwargs)(conv8)
    conv8 = Add()([skipconv8, conv8])
    conv8 = Activation('relu')(conv8)

    up9 = Conv2DTranspose(256, (2, 2), **conv_transpose_kwargs)(conv8)
    merge9 = concatenate([conv1, up9], axis=3)
    skipconv9 = Conv2D(512, 1, **conv_skip_kwargs)(merge9)
    conv9 = Conv2D(512, kernel_size, **conv_kwargs)(merge9)
    conv9 = Conv2D(512, kernel_size, **conv_kwargs)(conv9)
    conv9 = Add()([skipconv9, conv9])
    conv9 = Activation('relu')(conv9)

    # Correct dimensions
    conv9 = Conv2D(num_classes, 1, **conv_kwargs_fin)(conv9)
    activation = Softmax()(conv9)

    unet = Model(inputs=[inputs], outputs=[activation])
    unet.compile(optimizer=Adam(lr=lr), loss=loss, metrics=metrics)
    if (pretrained_weights):
        unet.load_weights(pretrained_weights)
    return unet

def multi_stream_unet_end_to_end(input_size, lr, metrics, num_classes):
    '''
    Function to build U-Net
    :param input_size: A tuple that specifies the input shape to the network
    :param num_classes: How many classes the classification problem at hand has
    :param lr: learning rate to use during optimization
    :param metrics: a list of the metrics to use
    :return: Returns a compiled U-Net
    '''
    conv_kwargs = {
        'strides': (1, 1),
        'padding': 'same',
        'activation': 'relu',
        'kernel_initializer': 'he_normal',
        'kernel_regularizer': l2(0.001)
    }
    conv_transpose_kwargs = {
        'strides': (2, 2),
        'kernel_initializer': 'he_normal',
        'kernel_regularizer': l2(0.001)
    }
    conv_kwargs_fin = {
        'strides': (1, 1),
        'padding': 'same',
        'activation': 'relu',
        'kernel_initializer': 'he_normal',
        'kernel_regularizer': l2(0.001)
    }
    pooling_kwargs = {
        'pool_size': (2, 2)
    }
    num_channels = input_size[-1]
    input = Input(shape=input_size)

    branch_outputs = []
    for i in range(num_channels):
        # Slicing the ith channel:
        in_ = Lambda(lambda x: x[:, :, :, i:(i+1)])(input)

        # Setting up your per-channel layers (replace with actual sub-models):
        conv1 = Conv2D(32, (3, 3), input_shape=(176, 176, 1), **conv_kwargs)(in_)
        conv1 = Conv2D(32, (3, 3), input_shape=(176, 176, 1), **conv_kwargs)(conv1)
        pool1 = MaxPooling2D(**pooling_kwargs)(conv1)

        conv2 = Conv2D(64, (3, 3), **conv_kwargs)(pool1)
        conv2 = Conv2D(64, (3, 3), **conv_kwargs)(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), **conv_kwargs)(pool2)
        conv3 = Conv2D(128, (3, 3), **conv_kwargs)(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), **conv_kwargs)(pool3)
        conv4 = Dropout(0.1)(conv4)
        conv4 = Conv2D(256, (3, 3), **conv_kwargs)(conv4)
        conv4 = Dropout(0.1)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), padding='same')(pool4)
        conv5 = Dropout(0.1)(conv5)
        conv5 = Conv2D(512, (3, 3), padding='same')(conv5)
        conv5 = Dropout(0.1)(conv5)

        up6 = Conv2DTranspose(128, (2, 2), **conv_transpose_kwargs)(conv5)
        merge6 = concatenate([conv4, up6], axis=3)
        conv6 = Conv2D(256, (3, 3), **conv_kwargs)(merge6)
        conv6 = Conv2D(256, (3, 3), **conv_kwargs)(conv6)

        up7 = Conv2DTranspose(128, (2, 2), **conv_transpose_kwargs)(conv6)
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(128, (3, 3), **conv_kwargs)(merge7)
        conv7 = Conv2D(128, (3, 3), **conv_kwargs)(conv7)

        up8 = Conv2DTranspose(128, (2, 2), **conv_transpose_kwargs)(conv7)
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(64, (3, 3), **conv_kwargs)(merge8)
        conv8 = Conv2D(64, (3, 3), **conv_kwargs)(conv8)

        up9 = Conv2DTranspose(128, (2, 2), **conv_transpose_kwargs)(conv8)
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, (3, 3), **conv_kwargs)(merge9)
        conv9 = Conv2D(64, (3, 3), **conv_kwargs)(conv9)

        branch_outputs.append(conv9)

    # Concatenating together the per-channel results:
    out = Concatenate()(branch_outputs)

    # Adding some further layers (replace or remove with your architecture):
    out = Conv2D(256, (3, 3), **conv_kwargs)(out)
    out = Conv2D(256, (3, 3), **conv_kwargs)(out)
    out = Conv2D(num_classes, (1, 1), **conv_kwargs)(out)
    out = Activation('softmax')(out)

    # Building model:
    model = Model(inputs=input, outputs=out)
    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=metrics)
    return model

def multi_stream_unet_trained_sequentially(m_t1, m_t1ce, m_t2, m_flair, lr, loss, metrics):
    '''
    :param m_t1: Compiled "unet_dong_each_mod" U-Net with weights trained on T1 modality
    :param m_t1ce:  Compiled "unet_dong_each_mod" U-Net with weights trained on T1ce modality
    :param m_t2: Compiled "unet_dong_each_mod" U-Net with weights trained on T2 modality
    :param m_flair: Compiled "unet_dong_each_mod" U-Net with weights trained on flair modality
    :param lr: Learning rate to be used during optimization
    :param loss: Loss function to be used during optimization
    :param metrics: A list of the metrics to be saved during optimization
    :return:
    '''
    kernel_size = 3
    conv_kwargs = {
        'strides': (1, 1),
        'padding': 'same',
        'activation': 'relu',
        'kernel_initializer': 'he_normal',
        'kernel_regularizer': l2(0.001)
    }
    conv_transpose_kwargs = {
        'strides': (2, 2),
        'kernel_initializer': 'he_normal',
        'kernel_regularizer': l2(0.001)
    }
    conv_kwargs_fin = {
        'strides': (1, 1),
        'padding': 'same',
        'activation': 'relu',
        'kernel_initializer': 'he_normal',
        'kernel_regularizer': l2(0.001)
    }
    pooling_kwargs = {
        'pool_size': (2, 2),
        'padding': 'valid'
    }
    
    m2_t1 = Model(m_t1.input, m_t1.layers[-3].output)
    m2_t1.trainable = False
    m2_t1ce = Model(m_t1ce.input, m_t1ce.layers[-3].output)
    m2_t1ce.trainable = False
    m2_t2 = Model(m_t2.input, m_t2.layers[-3].output)
    m2_t2.trainable = False
    m2_flair = Model(m_flair.input, m_flair.layers[-3].output)
    m2_flair.trainable = False
    
    input_ = Input((176, 176, 4))
    split = Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=3))(input_)
    
    output_t1 = m2_t1(split[0])
    output_t1ce = m2_t1ce(split[1])
    output_t2 = m2_t2(split[2])
    output_flair = m2_flair(split[3])
    
    tot_output = Concatenate(axis=-1)([output_t1, output_t1ce, output_t2, output_flair])
    
    out = Conv2D(256, (3, 3), **conv_kwargs)(tot_output)
    out = Conv2D(256, (3, 3), **conv_kwargs)(out)
    out = Conv2D(4, (1, 1), **conv_kwargs)(out)
    out = Activation('softmax')(out)
    
    model = Model(input=input_, output=out)
    model.compile(optimizer=Adam(lr=lr), loss=loss, metrics=metrics)
    return model
