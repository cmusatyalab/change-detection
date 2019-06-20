
import tensorflow as tf
from tensorflow import keras
import os
#import skimage.io as io
#import skimage.transform
import numpy as np
from tensorflow.keras import layers
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.utils import plot_model

WEIGHTS_PATH = "./vgg16_weights_tf_dim_ordering_tf_kernels.h5"

def zeroNet(input_size = (50176,11) , num_classes=11):
    inputs1 = Input(input_size)
    # out = Reshape((num_classes, input_size[0]*input_size[1]))(inputs1)
    # out = Permute((2,1))(out)
    final_out = Activation("softmax")(inputs1)

    model = Model(inputs = inputs1, outputs = final_out)
    model.summary()

    return model

def changeNet_VGG2(input_size = (224,224,3) , num_classes=11):
    #(240,180,1)
    # Share conv weights
    inputs1 = Input(input_size)
    inputs2 = Input(input_size)

    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv11 = conv1(inputs1)
    conv12 = conv1(inputs2)

    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv11 = conv1(conv11)
    conv12 = conv1(conv12)

    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2)
    pool11 = pool1(conv11)
    pool12 = pool1(conv12)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv21 = conv2(pool11)
    conv22 = conv2(pool12)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv21 = conv2(conv21)
    conv22 = conv2(conv22)

    
    pool2 = MaxPooling2D(pool_size=(2, 2),strides=2)
    pool21 = pool2(conv21)
    pool22 = pool2(conv22)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv31 = conv3(pool21)
    conv32 = conv3(pool22)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv31 = conv3(conv31)
    conv32 = conv3(conv32)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv31 = conv3(conv31)
    conv32 = conv3(conv32)

    pool3 = MaxPooling2D(pool_size=(2, 2), strides=2)
    pool31 = pool3(conv31)
    pool32 = pool3(conv32)
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv41 = conv4(pool31)
    conv42 = conv4(pool32)
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv41 = conv4(conv41)
    conv42 = conv4(conv42)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv41 = conv4(conv41)
    conv42 = conv4(conv42)

    pool4 = MaxPooling2D(pool_size=(2, 2), strides=2)
    pool41 = pool4(conv41)
    pool42 = pool4(conv42)

    conv4a = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv41 = conv4a(pool41)
    conv42 = conv4a(pool42)

    conv4a = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv41 = conv4a(conv41)
    conv42 = conv4a(conv42)

    conv4a = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv41 = conv4a(conv41)
    conv42 = conv4a(conv42)

    drop4 = Dropout(0.5)
    drop41 = drop4(conv41)
    drop42 = drop4(conv42)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv51 = conv5(pool41)
    conv52 = conv5(pool42)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv51 = conv5(conv51)
    drop51 = Dropout(0.5)(conv51)
    conv52 = conv5(conv52)
    drop52 = Dropout(0.5)(conv52)


    # Deconv Image 1
    up61 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop51)
    merge61 = concatenate([drop41,up61], axis = 3)
    conv61 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge61)
    conv61 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv61)

    up71 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (4,4))(conv61))
    merge71 = concatenate([conv31,up71], axis = 3)
    conv71 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge71)
    conv71 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv71)

    up81 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv71))
    merge81 = concatenate([conv21,up81], axis = 3)
    conv81 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge81)
    conv81 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv81)

    up91 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv81))
    merge91 = concatenate([conv11,up91], axis = 3)
    conv91 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge91)
    conv91 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv91)
    conv91 = Conv2D(num_classes, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv91)
    #conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    # Deconv Image 2
    up62 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop52)
    merge62 = concatenate([drop42,up62], axis = 3)
    conv62 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge62)
    conv62 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv62)

    up72 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (4,4))(conv62))
    merge72 = concatenate([conv32,up72], axis = 3)
    conv72 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge72)
    conv72 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv72)

    up82 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv72))
    merge82 = concatenate([conv22,up82], axis = 3)
    conv82 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge82)
    conv82 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv82)

    up92 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv82))
    merge92 = concatenate([conv12,up92], axis = 3)
    conv92 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge92)
    conv92 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv92)
    conv92 = Conv2D(num_classes, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv92)
    #conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    # Merge Deconv layers
    merged = concatenate([conv91,  conv92], axis=3)
    conv10 = Conv2D(num_classes, 1)(merged)
    out = Reshape((input_size[0]*input_size[1],num_classes))(conv10)
    final_out = Activation("softmax")(out)

    model = Model(inputs = [inputs1, inputs2], outputs = final_out)

    
    model.summary()

    #if(pretrained_weights):
    #	model.load_weights(pretrained_weights)
    plot_model(model, to_file='model.png')
    return model

def changeNet_VGG(input_size = (224,224,3) , num_classes=11):
    #(240,180,1)
    # Share conv weights
    inputs1 = Input(input_size)
    inputs2 = Input(input_size)

    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv11 = conv1(inputs1)
    conv12 = conv1(inputs2)

    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv11 = conv1(conv11)
    conv12 = conv1(conv12)

    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2)
    pool11 = pool1(conv11)
    pool12 = pool1(conv12)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv21 = conv2(pool11)
    conv22 = conv2(pool12)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv21 = conv2(conv21)
    conv22 = conv2(conv22)

    
    pool2 = MaxPooling2D(pool_size=(2, 2),strides=2)
    pool21 = pool2(conv21)
    pool22 = pool2(conv22)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv31 = conv3(pool21)
    conv32 = conv3(pool22)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv31 = conv3(conv31)
    conv32 = conv3(conv32)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv31 = conv3(conv31)
    conv32 = conv3(conv32)

    pool3 = MaxPooling2D(pool_size=(2, 2), strides=2)
    pool31 = pool3(conv31)
    pool32 = pool3(conv32)
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv41 = conv4(pool31)
    conv42 = conv4(pool32)
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv41 = conv4(conv41)
    conv42 = conv4(conv42)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv41 = conv4(conv41)
    conv42 = conv4(conv42)

    pool4 = MaxPooling2D(pool_size=(2, 2), strides=2)
    pool41 = pool4(conv41)
    pool42 = pool4(conv42)

    conv4a = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv41 = conv4a(pool41)
    conv42 = conv4a(pool42)

    conv4a = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv41 = conv4a(conv41)
    conv42 = conv4a(conv42)

    conv4a = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv41 = conv4a(conv41)
    conv42 = conv4a(conv42)

    drop4 = Dropout(0.5)
    drop41 = drop4(conv41)
    drop42 = drop4(conv42)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv51 = conv5(pool41)
    conv52 = conv5(pool42)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv51 = conv5(conv51)
    drop51 = Dropout(0.5)(conv51)
    conv52 = conv5(conv52)
    drop52 = Dropout(0.5)(conv52)


    # Deconv Image 1
    up61 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop51)
    merge61 = concatenate([drop41,up61], axis = 3)
    conv61 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge61)
    conv61 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv61)

    up71 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (4,4))(conv61))
    merge71 = concatenate([conv31,up71], axis = 3)
    conv71 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge71)
    conv71 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv71)

    up81 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv71))
    merge81 = concatenate([conv21,up81], axis = 3)
    conv81 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge81)
    conv81 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv81)

    up91 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv81))
    merge91 = concatenate([conv11,up91], axis = 3)
    conv91 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge91)
    conv91 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv91)
    conv91 = Conv2D(num_classes, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv91)
    #conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    # Deconv Image 2
    up62 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop52)
    merge62 = concatenate([drop42,up62], axis = 3)
    conv62 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge62)
    conv62 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv62)

    up72 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (4,4))(conv62))
    merge72 = concatenate([conv32,up72], axis = 3)
    conv72 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge72)
    conv72 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv72)

    up82 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv72))
    merge82 = concatenate([conv22,up82], axis = 3)
    conv82 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge82)
    conv82 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv82)

    up92 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv82))
    merge92 = concatenate([conv12,up92], axis = 3)
    conv92 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge92)
    conv92 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv92)
    conv92 = Conv2D(num_classes, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv92)
    #conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    # Merge Deconv layers
    merged = concatenate([conv91,  conv92], axis=3)
    conv10 = Conv2D(num_classes, 1)(merged)
    out = Reshape((num_classes, input_size[0]*input_size[1]))(conv10)
    out = Permute((2,1))(out)
    final_out = Activation("softmax")(out)

    model = Model(inputs = [inputs1, inputs2], outputs = final_out)

    
    model.summary()

    #if(pretrained_weights):
    #	model.load_weights(pretrained_weights)
    plot_model(model, to_file='model.png')
    return model

def changeNet(input_size = (256,256,3) , num_classes=11):
    #(240,180,1)
    # Share conv weights
    inputs1 = Input(input_size)
    inputs2 = Input(input_size)

    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv11 = conv1(inputs1)
    conv12 = conv1(inputs2)

    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv11 = conv1(conv11)
    conv12 = conv1(conv12)

    pool1 = MaxPooling2D(pool_size=(2, 2))
    pool11 = pool1(conv11)
    pool12 = pool1(conv12)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv21 = conv2(pool11)
    conv22 = conv2(pool12)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv21 = conv2(conv21)
    conv22 = conv2(conv22)

    
    pool2 = MaxPooling2D(pool_size=(2, 2))
    pool21 = pool2(conv21)
    pool22 = pool2(conv22)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv31 = conv3(pool21)
    conv32 = conv3(pool22)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv31 = conv3(conv31)
    conv32 = conv3(conv32)

    pool3 = MaxPooling2D(pool_size=(2, 2))
    pool31 = pool3(conv31)
    pool32 = pool3(conv32)
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv41 = conv4(pool31)
    conv42 = conv4(pool32)
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv41 = conv4(conv41)
    conv42 = conv4(conv42)


    drop4 = Dropout(0.5)
    drop41 = drop4(conv41)
    drop42 = drop4(conv42)

    pool4 = MaxPooling2D(pool_size=(2, 2))
    pool41 = pool4(drop41)
    pool42 = pool4(drop42)
    
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv51 = conv5(pool41)
    conv52 = conv5(pool42)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv51 = conv5(conv51)
    drop51 = Dropout(0.5)(conv51)
    conv52 = conv5(conv52)
    drop52 = Dropout(0.5)(conv52)

    # Deconv Image 1 

    up61 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop51))
    merge61 = concatenate([drop41,up61], axis = 3)
    conv61 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge61)
    conv61 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv61)

    up71 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv61))
    merge71 = concatenate([conv31,up71], axis = 3)
    conv71 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge71)
    conv71 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv71)

    up81 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv71))
    merge81 = concatenate([conv21,up81], axis = 3)
    conv81 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge81)
    conv81 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv81)

    up91 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv81))
    merge91 = concatenate([conv11,up91], axis = 3)
    conv91 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge91)
    conv91 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv91)
    conv91 = Conv2D(num_classes, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv91)
    #conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    # Deconv Image 2
    up62 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop52))
    merge62 = concatenate([drop42,up62], axis = 3)
    conv62 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge62)
    conv62 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv62)

    up72 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv62))
    merge72 = concatenate([conv32,up72], axis = 3)
    conv72 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge72)
    conv72 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv72)

    up82 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv72))
    merge82 = concatenate([conv22,up82], axis = 3)
    conv82 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge82)
    conv82 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv82)

    up92 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv82))
    merge92 = concatenate([conv12,up92], axis = 3)
    conv92 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge92)
    conv92 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv92)
    conv92 = Conv2D(num_classes, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv92)
    #conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    # Merge Deconv layers
    merged = concatenate([conv91,  conv92], axis=3)
    conv10 = Conv2D(num_classes, 1)(merged)

    out = Reshape((num_classes, input_size[0]*input_size[1]))(conv10)
    out = Permute((2,1))(out)
    final_out = Activation("softmax")(out)

    model = Model(inputs = [inputs1, inputs2], outputs = final_out)

    
    model.summary()

    #if(pretrained_weights):
    #	model.load_weights(pretrained_weights)
    plot_model(model, to_file='model.png')
    return model
def VGG16(include_top=True, weights='imagenet',
          input_tensor=None, input_shape=None,
          pooling=None,
          classes=1000):
    """Instantiates the VGG16 architecture.
    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 244)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
    # Determine proper input shape
    input_shape = (224,224,3)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    
    # Block 1
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(4096, activation='relu', name='fc1')(x)
        x = layers.Dense(4096, activation='relu', name='fc2')(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    
    inputs = img_input
    # Create model.
    model = tf.keras.models.Model(inputs, x, name='vgg16')

    # load weights
    # if weights == 'imagenet':
    #     if include_top:
    #         weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
    #                                 WEIGHTS_PATH,
    #                                 cache_subdir='models')
    #     else:
    #         weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
    #                                 WEIGHTS_PATH_NO_TOP,
    #                                 cache_subdir='models')
    model.load_weights(WEIGHTS_PATH)
        
    return model

if __name__=="__main__":
    model = changeNet()
    #model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
    #model.fit_generator(myGene,steps_per_epoch=300,epochs=1,callbacks=[model_checkpoint])

    #testGene = testGenerator("data/membrane/test")
    #results = model.predict_generator(testGene,30,verbose=1)
    #saveResult("data/membrane/test",results)