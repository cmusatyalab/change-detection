import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
from skimage import io
import argparse
import matplotlib.pyplot as plt
import pickle
import time
import h5py
import glob
import os
import time
import numpy.random
import pdb
#from pdb import set_trace as st

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
                          OtherObjects, PersonCycle, Rubbish, Sign, TrafficCone, Vehicle])
CHANGE_CLASSES = ["NoChange", "Barrier", "Bin", "ConstructionMaintenance",
                          "OtherObjects", "PersonCycle", "Rubbish", "Sign", "TrafficCone", "Vehicle"]

''' Returns 224 x 224 x 1 vector '''
def prepareGTData(mask, num_class = 10):
    #pdb.set_trace()
    mask = mask.astype('int')
    new_mask = np.zeros(mask.shape[0:2], dtype=np.int64)
    # Mask Out Class
    new_mask[np.all(mask == np.zeros((1,1,3), dtype=np.int32) , axis=2)] = np.int64(0)
    new_mask[np.all(mask == np.array(Misc, dtype=np.int32).reshape(1,1,3) , axis=2)] = np.int64(4)
    OtherObjects
    # All other classes
    for i in range(num_class):
        new_mask[np.all(mask == CLASS_DICT[i].reshape(1,1,3) , axis=2)] = np.int64(i)
    
    return new_mask

class CityscapesSegmentation(Dataset):
    
    def __init__(self, args=None, root='./cityscapes', split="train", training=False, probability=0.5, image_size=(224,224)):

        self.root = root
        self.split = split
        self.args = args
        self.files = {}
        self.training = training
        self.image_size = image_size
        self.p = probability
        self.num_classes = 10
        self.images_base = os.path.join(self.root, 'leftImg8bit', self.split)
        self.annotations_base = os.path.join(self.root, 'gtFine', self.split)

        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.png')

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 15, 16, 18, 23, -1]
        self.valid_classes = [ [12,13], [7], [11,14],[22],[33,24,25,32],[8],[19,20,17],[21],[26,27,28,29,30,31]]
        self.class_names = ['wall-fence', 'road', 'building-guardrail', 'terrain', 'bicycle-person-motorcycle-rider',
                            'sidewalk', 'trafficsign-trafficlight-pole', 'vegetation','car-truck-bus-train']
        

        self.ignore_index = 0
        #self.class_map = dict(zip(self.valid_classes, range(self.num_classes)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

        self.resize_image = transforms.Compose([
            transforms.ToPILImage(), 
            transforms.Resize(self.image_size),
        ])
        self.max_rot = 15
        self.max_trans = 0.2
        self.max_scale_diff = 0.2
        self.max_shear_deg = 10

        self.training = training
        self.p = probability
        self.normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(self.annotations_base,
                                img_path.split(os.sep)[-2],
                                os.path.basename(img_path)[:-15] + 'gtFine_labelIds.png')
        

        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        gt = self.decode_seg(_tmp)
        #pdb.set_trace()
        gt = Image.fromarray(gt)
        

        im = io.imread(img_path)
        im = self.resize_image(im).convert('RGB')
        gt = transforms.Resize(self.image_size)(gt)

        if self.training:
            # Shear
            if (np.random.rand() < self.p):
                angle = int( self.max_rot * np.random.rand() )
                translation = (self.max_trans * np.random.rand() , self.max_trans * np.random.rand() )
                scale = self.max_scale_diff * 2. * (np.random.rand() - 0.5) + 1.
                shear = int( self.max_shear_deg * np.random.rand() )
                im = transforms.functional.affine(im, angle=angle, translate=translation, 
                                                   scale=scale, shear=shear, resample=Image.NEAREST, fillcolor=None)
                gt = transforms.functional.affine(gt, angle=angle, translate=translation, 
                                                   scale=scale, shear=shear, resample=Image.NEAREST, fillcolor=None)
            if (np.random.rand() < self.p):
                im = transforms.functional.hflip(im)
                gt = transforms.functional.hflip(gt)


        im = self.normalize_transform( transforms.ToTensor()(im).float() )

        #gt_orig = transforms.ToTensor()(gt)
        #gt = prepareGTData( np.moveaxis(gt_orig.numpy()*255, 0, -1))
        #pdb.set_trace()
        gt = 255.*transforms.ToTensor()(gt)
        gte = gt.numpy().squeeze()
        edges_x = np.zeros(self.image_size)
        edges_y = np.zeros(self.image_size)
        edges_x[:-1,:] = np.abs(gte[1:,:]-gte[:-1,:])
        edges_y[:,:-1] = np.abs(gte[:,1:]-gte[:,:-1])
        edges = edges_x + edges_y
        edges = edges != 0
        edges[:,0:1] = 0
        edges[:,-2:-1] = 0
        edges[0:1,:] = 0
        edges[-2:-1,:] = 0
        #pdb.set_trace()
        edges = torch.from_numpy(edges.astype(np.float32))
        gt = gt.long()
        return { 'im': im, 'gt': gt, 'edges':edges }

    def decode_seg(self, mask):
        temp = mask
        for _voidc in self.void_classes:
            mask[temp == _voidc] = np.int64(self.ignore_index)
        for i, classes in enumerate(self.valid_classes):
            #pdb.set_trace()
            for _validc in classes:
                mask[temp == _validc] = np.int64(i+1) #self.class_map[_validc]
        return mask

    def recursive_glob(self, rootdir='.', suffix=''):
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

def parse_args():
    parser = argparse.ArgumentParser(description='Dataset Loader for PyTorch Networks')
    parser.add_argument('--root', type=str, default='./train/train.txt')
    parser.add_argument('--ann_path', type=str, default='./train/train.txt')
    return parser.parse_args()



def visualizeCityImages(im1,mask):
    mean = torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape(3,1,1)).float()
    std = torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape(3,1,1)).float()
    transform = transforms.ToPILImage()
    
    c, h, w = im1.shape
    combined = np.zeros((c, h, w*2))
    combined[:,:, :w] = std*im1 + mean
    combined[:,:, w:2*w] = mask
    combined = transform(torch.from_numpy(combined).float())
    # im1 = transform(std*im1 + mean)
    # im2 = transform(std*im2 + mean)
    return combined

def visualizeAllCityImages(im1,gt,mask):
    mean = torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape(3,1,1)).float()
    std = torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape(3,1,1)).float()
    transform = transforms.ToPILImage()
    
    c, h, w = im1.shape
    combined = np.zeros((c, h, w*3))
    combined[:,:, :w] = std*im1 + mean
    combined[:,:, w:2*w] = gt
    combined[:,:, 2*w:3*w] = mask
    combined = transform(torch.from_numpy(combined).float())
    # im1 = transform(std*im1 + mean)
    # im2 = transform(std*im2 + mean)
    return combined

def labelVisualize(img, num_class=10, target_size = (224,224)):
    #img = (img.reshape( num_class, target_size[0], target_size[1]) ).argmax( axis=0 )
    seg_img = np.zeros( ( target_size[0], target_size[1], 3 ) ).astype('uint8')
    for c in range(num_class):
        #pdb.set_trace()
        seg_img[:,:,0] += ( (img[:,: ] == c )*( CLASS_DICT[c][0] )).astype('uint8')
        seg_img[:,:,1] += ((img[:,: ] == c )*( CLASS_DICT[c][1] )).astype('uint8')
        seg_img[:,:,2] += ((img[:,: ] == c )*( CLASS_DICT[c][2] )).astype('uint8')
        
    #seg_img = io.transform.resize(seg_img  , (768, 1024 ))
    #skio.imsave(  "output/output" + str(imNumber) + ".png" , seg_img )
    return seg_img

#def main(args):
    

    # vlcmu = CityscapesSegmentation(args.train_file_path, training=True, probability=0.5)
    # for idx in range(len(vlcmu)):
    #     print(idx)
    #     data = vlcmu[idx]
    #     print(data['gt'].shape)
    #     plt.imshow(visualizeImages(data['im1'],data['im2'],data['gt_orig']))
    #     plt.draw()
    #     plt.pause(0.3)


if __name__ == '__main__':
    
    # from dataloaders.utils import decode_segmap
    # from torch.utils.data import DataLoader
    # import matplotlib.pyplot as plt
    # import argparse

    args = parse_args()
    #main(args)


    cityscapes_train = CityscapesSegmentation(args, split='train')

    for idx in range(len(cityscapes_train)):
        print(idx)
        data = cityscapes_train[idx]
        print(data['gt'].shape)
        #pdb.set_trace()
        #plt.imshow(visualizeCityImages(data['im'],np.moveaxis(labelVisualize( data['gt'].reshape(224,224).numpy() ), -1, 0)))
        plt.imshow( data['gt'].reshape(224,224).numpy() )
        #edges = data['edges']
        plt.imshow(edges)
        plt.draw()
        plt.pause(0.3)