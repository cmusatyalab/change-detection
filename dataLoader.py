import tensorflow as tf
from tensorflow import keras
from tensorflow.examples.tutorials.mnist import input_data

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization, Conv2DTranspose
from keras.models import Sequential
import numpy as np
import numpy.random
import skimage.io as skio
import time
from io import StringIO
import os
from shutil import copy
from itertools import izip, count
from keras.utils.np_utils import to_categorical
import skimage.transform
import pdb
#import cv2
np.set_printoptions(threshold=np.inf)

#   From dataset, we have the following definitions
#
#   classcolors = [0 0 0; % ignore        0 'mask-out' (ignore)
#                255 255 255; % unchanged 1 'no-change'
#                136 0 21; % brown/red    2 'barrier'
#                237 28 36; % red         3 'bin'
#                255 127 39; % orange     4 'construction-maintenance'
#                255 242 0; % yellow      5 'misc'
#                34 177 76; % dark green  6 'other-objects'
#                0 162 232; % light blue  7 'person-cycle'
#                63 72 204; % navy blue   8 'rubbish'
#                163 73 164; % purple     9 'sign'
#                255 174 201; % pink      10 'traffic-cone'
#                181 230 29]; % lime      11 'vehicle'

MaskOut = [0,0,0]
NoChange = [255, 255, 255]
Barrier = [136, 0, 21]
Bin = [237, 28, 36]
ConstructionMaintenance = [255, 127, 39]
Misc = [255, 242, 0]
OtherObjects = [34, 177, 76]
PersonCycle = [0, 162, 232]
Rubbish = [63, 72, 204]
Sign = [163, 73, 164]
TrafficCone = [255, 174, 201]
Vehicle = [181, 230, 29]

CLASS_DICT = np.array([NoChange, Barrier, Bin, ConstructionMaintenance,
                          Misc, OtherObjects, PersonCycle, Rubbish, Sign, TrafficCone, Vehicle])
CHANGE_CLASSES = ["NoChange", "Barrier", "Bin", "ConstructionMaintenance",
                          "Misc", "OtherObjects", "PersonCycle", "Rubbish", "Sign", "TrafficCone", "Vehicle"]

def readSplitFile(fileName):
    numbers = np.loadtxt(fileName)
    split = []
    for n in numbers:
        if(n < 10):
            split.append("00" + str(int(n)))
        elif(n < 100):
            split.append("0" + str(int(n)))
        else:
            split.append(str(int(n)))
    return split


def readImage(path):
    return np.loadtxt(path)

def prepareData(img1, img2, mask, num_class = 11):
    output = np.zeros(mask.shape[0:3] + (num_class,))
    for b in range(0,mask.shape[0]):
        new_mask = np.zeros(mask[b,:,:,0].shape + (num_class,))
        bmask = mask[b,:,:,:]
        for i in range(num_class):
            #convert to one-hot matrix
            #select_mask = np.all([ [bmask[:,:,0] == CLASS_DICT[i][0] ] , [ bmask[:,:,1] == CLASS_DICT[i][1] ] , [bmask[:,:,2] == CLASS_DICT[i][2] ] ] , axis= 2)
            #select_mask = np.all([bmask[:,:,0] == CLASS_DICT[i][0] , bmask[:,:,1] == CLASS_DICT[i][1] , bmask[:,:,2] == CLASS_DICT[i][2] ] , axis= 0)
            #new_mask[select_mask,i] = 1
            #select_mask = np.all(bmask == CLASS_DICT[i].reshape(1,1,3) , axis=2)
            #pdb.set_trace()
            new_mask[np.all(bmask == CLASS_DICT[i].reshape(1,1,3) , axis=2),i] = 1
        output[b,:,:,:] = new_mask
    output = np.reshape(output,(output.shape[0],output.shape[1]*output.shape[2],output.shape[3]))
    return (img1,img2,output)

def prepareGTData(mask, num_class = 11):
    output = np.zeros(mask.shape[0:3] + (num_class,))
    for b in range(0,mask.shape[0]):
        new_mask = np.zeros(mask[b,:,:,0].shape + (num_class,))
        bmask = mask[b,:,:,:]
        for i in range(num_class):
            #convert to one-hot matrix
            #select_mask = np.all([ [bmask[:,:,0] == CLASS_DICT[i][0] ] , [ bmask[:,:,1] == CLASS_DICT[i][1] ] , [bmask[:,:,2] == CLASS_DICT[i][2] ] ] , axis= 2)
            #select_mask = np.all([bmask[:,:,0] == CLASS_DICT[i][0] , bmask[:,:,1] == CLASS_DICT[i][1] , bmask[:,:,2] == CLASS_DICT[i][2] ] , axis= 0)
            #new_mask[select_mask,i] = 1
            #select_mask = np.all(bmask == CLASS_DICT[i].reshape(1,1,3) , axis=2)
            #pdb.set_trace()
            new_mask[np.all(bmask == CLASS_DICT[i].reshape(1,1,3) , axis=2),i] = 1
        output[b,:,:,:] = new_mask
    output = np.reshape(output,(output.shape[0],output.shape[1]*output.shape[2],output.shape[3]))
    return output

def prepareGTNoChangeData(mask, num_class = 11):
    output = np.zeros(mask.shape[0:3] + (num_class,))
    for b in range(0,mask.shape[0]):
        new_mask = np.zeros(mask[b,:,:,0].shape + (num_class,))
        new_mask[:,:,0] = 1
        output[b,:,:,:] = new_mask
    output = np.reshape(output,(output.shape[0],output.shape[1]*output.shape[2],output.shape[3]))
    return output


def getDataList(dataset_path, split):
    image1_list = []
    image2_list = []
    gt_list = []
    for tfile in split:
        for gt in sorted(os.listdir(dataset_path + tfile + "/GT")):
            if(gt[-4:] == ".png"):
                gt_list.append(dataset_path + tfile + "/GT/" + gt)                
                #print(gt)
        for imfile in sorted(os.listdir(dataset_path + tfile + "/RGB/")):
            if(imfile[-4:] == ".png" and imfile[0] == "1"):
                image1_list.append(dataset_path + tfile + "/RGB/" + imfile)                
                #print(imfile)
            elif(imfile[-4:] == ".png" and imfile[0] == "2"):
                #print(imfile)
                image2_list.append(dataset_path + tfile + "/RGB/" + imfile)   
    
    return image1_list, image2_list, gt_list

def loadDataListsIntoFolders(img1_list,img2_list, gt_list, direct = "train/", flag_multi_class = True, num_class = 11):
    #comb_img = []
    #image_arr1 = []
    #image_arr2 = []
    #mask_arr = []
    #comb_mask = []
    for i, _ in enumerate(img1_list):
        # img1 = skio.imread(img1_list[i])
        # img2 = skio.imread(img2_list[i])
        # mask = skio.imread(gt_list[i])
        copy(img1_list[i], direct + "im1/1_im" + str(i) + ".png")
        copy(img2_list[i], direct + "im2/2_im" + str(i) + ".png")
        copy(gt_list[i], direct + "gt/gt" + str(i) + ".png")
        #img1,img2,mask = prepareData(img1, img2, mask)
        #image_arr1.append(img1)
        #image_arr2.append(img2)
        #mask_arr.append(mask)
        #print(i)
    #image_1 = np.array(image_arr1)
    #image_2 = np.array(image_arr2)
    #mask_a = np.array(mask_arr)
    return #image_1, image_2, mask_a

def generateTrainValDataGenerator(batch_size,train_path,image1_folder, image2_folder, mask_folder, aug_dict_im, aug_dict_mask, image_color_mode = "rgb",
                    mask_color_mode = "rgb",image_save_prefix1  = "1_image",image_save_prefix2  = "2_image", mask_save_prefix  = "mask",
                    flag_multi_class = True,num_class = 11,save_to_dir = "/home/kchrist/change_detection/output",target_size = (256,256),seed = 1):
    
    image1_datagen = ImageDataGenerator(**aug_dict_im)
    image2_datagen = ImageDataGenerator(**aug_dict_im)
    mask_datagen = ImageDataGenerator(**aug_dict_mask)

    image1_generator_train = image1_datagen.flow_from_directory(
        directory=train_path,
        classes = [image1_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix1,
        subset='training',
        seed = seed)
    image2_generator_train = image2_datagen.flow_from_directory(
        directory=train_path,
        classes = [image2_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix2,
        subset='training',
        seed = seed)
    mask_generator_train = mask_datagen.flow_from_directory(
        directory=train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        subset='training',
        seed = seed)
    # image1_generator_val = image1_datagen.flow_from_directory(
    #     directory=train_path,
    #     classes = [image1_folder],
    #     class_mode = None,
    #     color_mode = image_color_mode,
    #     target_size = target_size,
    #     batch_size = batch_size,
    #     save_to_dir = save_to_dir,
    #     save_prefix  = image_save_prefix1,
    #     subset='validation',
    #     seed = seed)
    # image2_generator_val = image2_datagen.flow_from_directory(
    #     directory=train_path,
    #     classes = [image2_folder],
    #     class_mode = None,
    #     color_mode = image_color_mode,
    #     target_size = target_size,
    #     batch_size = batch_size,
    #     save_to_dir = save_to_dir,
    #     save_prefix  = image_save_prefix2,
    #     subset='validation',
    #     seed = seed)
    # mask_generator_val = mask_datagen.flow_from_directory(
    #     directory=train_path,
    #     classes = [mask_folder],
    #     class_mode = None,
    #     color_mode = mask_color_mode,
    #     target_size = target_size,
    #     batch_size = batch_size,
    #     save_to_dir = save_to_dir,
    #     save_prefix  = mask_save_prefix,
    #     subset='validation',
    #     seed = seed)

    train_generator = izip(image1_generator_train, image2_generator_train, mask_generator_train)
    #val_generator = izip(image1_generator_val, image2_generator_val, mask_generator_val)
    for img1,img2,mask in train_generator:
        #print("im herec")
        
        img1, img2, mask = prepareData(img1, img2, mask, num_class)
        #yield ({'input_1': img1, 'input_2': img2}, {'output': mask})
        # print(img1.shape)
        # print(img2.shape)
        # print(mask.shape)
        #skio.imsave('gt.png',img1)
        if (np.random.randint(100) < 50):
            img1,img2 = img2,img1
        
        yield [img1, img2],mask
    #return train_generator#, val_generator


def trainvalDataGenerator(generator, num_class=11):
    for img1,img2,mask in generator:
        #print("im herec")
        
        img1, img2, mask = prepareData(img1, img2, mask, num_class)
        #yield ({'input_1': img1, 'input_2': img2}, {'output': mask})
        # print(img1.shape)
        # print(img2.shape)
        # print(mask.shape)
        #skio.imsave('gt.png',img1)
        # if (np.random.randint(100) < 50):
        #     temp = img2
        #     img2 = img1
        #     img1 = temp
        
        yield [img1, img2],mask


def testDataGenerator(im1_path, im2_path, gt_path, target_size = (256,256),flag_multi_class = True):
    idx = 0
    for im1, im2, gt in izip(sorted(os.listdir(im1_path)) , sorted(os.listdir(im2_path)) , sorted(os.listdir(gt_path)) ):
        img1 = skio.imread(im1_path + "/" + im1)
        img1 = img1 / 255.
        img1 = skimage.transform.resize(img1,target_size)
        img2 = skio.imread(im2_path + "/" + im2)
        img2 = img2 / 255.
        img2 = skimage.transform.resize(img2,target_size)
        mask = skio.imread(gt_path + "/" + gt)
        skio.imsave('output/gt' + str(idx) + ".png",mask)
        img1 = np.reshape(img1,(1,)+img1.shape)
        img2 = np.reshape(img2,(1,)+img2.shape)
        idx+=1
        yield [img1,img2]

def testGroundTruthGenerator(im1_path, im2_path, gt_path, target_size = (256,256),flag_multi_class = True):
    idx = 0
    for gt in sorted(os.listdir(gt_path)):
        mask = skio.imread(gt_path + "/" + gt)
        mask = prepareGTData(mask, num_class = 11)
        idx+=1
        yield mask

def testEvaluationDataGenerator(test_generator, target_size = (256,256),flag_multi_class = True):
    for im1, im2, gt in test_generator:
        img1 = img1 / 255.
        img1 = skimage.transform.resize(img1,target_size)
        img2 = img2 / 255.
        img2 = skimage.transform.resize(img2,target_size)
        img1 = np.reshape(img1,(1,)+img1.shape)
        img2 = np.reshape(img2,(1,)+img2.shape)
        img1, img2, mask = prepareData(img1, img2, mask, num_class=11)
        yield [img1,img2], mask

def generateTestGenerator(batch_size,train_path,image1_folder, image2_folder, mask_folder, aug_dict_im, aug_dict_mask, image_color_mode = "rgb",
                    mask_color_mode = "rgb",image_save_prefix1  = "1_image",image_save_prefix2  = "2_image", mask_save_prefix  = "mask",
                    flag_multi_class = True,num_class = 11,save_to_dir = "/home/kchrist/change_detection/output",target_size = (256,256),seed = 1):
    
    image1_datagen = ImageDataGenerator(**aug_dict_im)
    image2_datagen = ImageDataGenerator(**aug_dict_im)
    mask_datagen = ImageDataGenerator(**aug_dict_mask)

    image1_generator_test = image1_datagen.flow_from_directory(
        directory=train_path,
        classes = [image1_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix1,
        shuffle=False,
        seed = seed)
    image2_generator_test = image2_datagen.flow_from_directory(
        directory=train_path,
        classes = [image2_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix2,
        shuffle=False,
        seed = seed)
    mask_generator_test = mask_datagen.flow_from_directory(
        directory=train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        shuffle=False,
        seed = seed)
    

    test_generator = izip(image1_generator_test, image2_generator_test, mask_generator_test)
    idx = 0
    for img1,img2,mask in test_generator:
        for i in range(img1.shape[0]):
            skio.imsave(  "output/im1_" + str(idx) + ".png" , skimage.transform.resize( (255.*img1[i,:,:,:]).astype('uint8') , (768, 1024)) )
            skio.imsave(  "output/im2_" + str(idx) + ".png" , skimage.transform.resize((255.*img2[i,:,:,:]).astype('uint8') , (768, 1024)) )
            skio.imsave(  "output/gt_" + str(idx) + ".png" , skimage.transform.resize(mask[i,:,:,:].astype('uint8') , (768, 1024))  )
            idx += 1    
        img1, img2, mask = prepareData(img1, img2, mask, num_class)
        yield [img1, img2],mask

def generateAllNoChangeTestGenerator(batch_size,train_path,image1_folder, image2_folder, mask_folder, aug_dict_im, aug_dict_mask, image_color_mode = "rgb",
                    mask_color_mode = "rgb",image_save_prefix1  = "1_image",image_save_prefix2  = "2_image", mask_save_prefix  = "mask",
                    flag_multi_class = True,num_class = 11,save_to_dir = "/home/kchrist/change_detection/output",target_size = (256,256),seed = 1):
    
    image1_datagen = ImageDataGenerator(**aug_dict_im)
    image2_datagen = ImageDataGenerator(**aug_dict_im)
    mask_datagen = ImageDataGenerator(**aug_dict_mask)

    image1_generator_test = image1_datagen.flow_from_directory(
        directory=train_path,
        classes = [image1_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix1,
        shuffle=False,
        seed = seed)
    image2_generator_test = image2_datagen.flow_from_directory(
        directory=train_path,
        classes = [image2_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix2,
        shuffle=False,
        seed = seed)
    mask_generator_test = mask_datagen.flow_from_directory(
        directory=train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        shuffle=False,
        seed = seed)
    

    test_generator = izip(image1_generator_test, image2_generator_test, mask_generator_test)
    idx = 0
    for img1,img2,mask in test_generator: 
        dup_mask = mask
        img1, img2, mask = prepareData(img1, img2, mask, num_class)
        allnochange = prepareGTNoChangeData(dup_mask, num_class)
        yield allnochange,mask

def labelVisualize(num_class,img, imNumber, target_size = (224,224)):
    img = (img.reshape( target_size[0], target_size[1] , num_class ) ).argmax( axis=-1 )
    seg_img = np.zeros( ( target_size[0], target_size[1] , 3  ) ).astype('uint8')
    for c in range(num_class):
        seg_img[:,:,0] += ( (img[:,: ] == c )*( CLASS_DICT[c][0] )).astype('uint8')
        seg_img[:,:,1] += ((img[:,: ] == c )*( CLASS_DICT[c][1] )).astype('uint8')
        seg_img[:,:,2] += ((img[:,: ] == c )*( CLASS_DICT[c][2] )).astype('uint8')
    seg_img = skimage.transform.resize(seg_img  , (768, 1024 ))
    skio.imsave(  "output/output" + str(imNumber) + ".png" , seg_img )
    # #img = img[:,:,0] if len(img.shape) == 3 else img
    # print(np.sum(img))
    # img = np.reshape(img,[256,256,11])
    
    # #img_out = np.zeros(img.shape + (3,))
    # img_out = np.zeros([256,256,3])#.astype(np.uint8)
    # for i in range(num_class):
    #     img_out[img[:,:,i] > 0.8,:] = 2.*CLASS_DICT[i]/255. -1.
    # return img_out

def saveResult(save_path,results,flag_multi_class = True,num_class = 11, target_size = (224,224)):
    for i,item in enumerate(results):
        #print(item)
        img = labelVisualize(num_class, item, i, target_size = (224,224))
        #print(img.shape)
        #skio.imsave(save_path + "predict" + str(i) + ".png",img)

def visualizeGTColors(num_class=11, target_size = (224,224)):
    seg_img = np.zeros( ( target_size[0], target_size[1] , 3  ) ).astype('uint8')
    for c in range(num_class):
        seg_img[:,:,0] = ( ( CLASS_DICT[c][0] )).astype('uint8')
        seg_img[:,:,1] = (( CLASS_DICT[c][1] )).astype('uint8')
        seg_img[:,:,2] = (( CLASS_DICT[c][2] )).astype('uint8')
        #seg_img = skimage.transform.resize(seg_img  , (768, 1024 ))
        skio.imsave(  "output/a" + str(c) + ".png" , seg_img )

if __name__=="__main__":
    visualizeGTColors()
    """
    test_split_path = "./ChangeDetection/test_split.txt"
    train_split_path = "./ChangeDetection/train_split.txt"
    dataset_path = "ChangeDetection/raw/"

    test_split = readSplitFile(test_split_path)
    train_split = readSplitFile(train_split_path)

    print(test_split)
    train_im1, train_im2, train_gt = getDataList(dataset_path, train_split)
    test_im1, test_im2, test_gt = getDataList(dataset_path, test_split)
    print(test_gt)
    loadDataListsIntoFolders(train_im1, train_im2, train_gt, "train/")
    loadDataListsIntoFolders(test_im1, test_im2, test_gt, "test/")
    """
