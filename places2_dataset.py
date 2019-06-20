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

import utils
import pdb
#from pdb import set_trace as st

class Places2DatasetCityScapesMasks(Dataset):
    ''''
    Dataset for Training Background Inpainting network
    '''
    def __init__(self, places2_filelist_path, places2_basepath, root='./cityscapes', split="train", training=False, probability=0.5, image_size = (256,256)):
        self.root=root
        self.split=split
        self.files = {}
        self.image_size = image_size
        with open(places2_filelist_path, 'r') as f:
            self.places2_filelist = f.read().splitlines()
        self.places2_basepath = places2_basepath

        # Cityscapes 
        self.images_base = os.path.join(self.root, 'leftImg8bit', self.split)
        self.annotations_base = os.path.join(self.root, 'gtFine', self.split)
        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.png')
        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

        self.ignore_index = 0

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 15, 16, 18, 23, -1]
        self.valid_classes = [ [12,13], [7], [11,14],[22],[33,24,25,32],[8],[19,20,17],[21],[26,27,28,29,30,31]]
        self.class_names = ['wall-fence', 'road', 'building-guardrail', 'terrain', 'bicycle-person-motorcycle-rider',
                            'sidewalk', 'trafficsign-trafficlight-pole', 'vegetation','car-truck-bus-train']

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
        # self.normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
        self.normalize_transform = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])


    def __len__(self):
        return len(self.places2_filelist)

    def __getitem__(self, idx):
        #pdb.set_trace()
        index = np.random.randint(len(self.files[self.split]))
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(self.annotations_base,
                                img_path.split(os.sep)[-2],
                                os.path.basename(img_path)[:-15] + 'gtFine_labelIds.png')
        instance_path = os.path.join(self.annotations_base,
                                img_path.split(os.sep)[-2],
                                os.path.basename(img_path)[:-15] + 'gtFine_instanceIds.png')

        
        _tmp = io.imread(instance_path)
        segmentIds = np.unique(_tmp)
        instance_id = np.random.randint(len(segmentIds))
        mask = _tmp == segmentIds[instance_id]
        mask = mask.astype(np.float32)
        mask = transforms.ToPILImage()(mask)
        mask = transforms.Resize(self.image_size)(mask)
        # gt = self.decode_seg(_tmp)
        # gt = Image.fromarray(gt)
        

        places2_impath = self.places2_basepath + self.places2_filelist[idx]

        im = io.imread(places2_impath)
        im = self.resize_image(im).convert('RGB')

        #im = torch.from_numpy(io.imread(places2_impath)).permute(2,0,1).float() / 255

        if self.training:
            # Shear
            if (np.random.rand() < self.p):
                angle = int( self.max_rot * np.random.rand() )
                translation = (self.max_trans * np.random.rand() , self.max_trans * np.random.rand() )
                scale = self.max_scale_diff * 2. * (np.random.rand() - 0.5) + 1.
                shear = int( self.max_shear_deg * np.random.rand() )
                im = transforms.functional.affine(im, angle=angle, translate=translation, 
                                                   scale=scale, shear=shear, resample=Image.NEAREST, fillcolor=None)
                mask = transforms.functional.affine(mask, angle=angle, translate=translation, 
                                                   scale=scale, shear=shear, resample=Image.NEAREST, fillcolor=None)
            if (np.random.rand() < self.p):
                im = transforms.functional.hflip(im)
                mask = transforms.functional.hflip(mask)


        im = self.normalize_transform( transforms.ToTensor()(im).float() )
        mask = transforms.ToTensor()(mask)
        #pdb.set_trace()
        ret = {'im':im,
               'mask':mask,
              }

        return ret

    def decode_seg(self, mask):
        temp = mask
        for _voidc in self.void_classes:
            mask[temp == _voidc] = np.int64(self.ignore_index)
        for i, classes in enumerate(self.valid_classes):
            #pdb.set_trace()
            for _validc in classes:
                mask[temp == _validc] = np.int64(i+1) #self.class_map[_validc]
        return mask

    def decode_instance(self, mask):
        #pdb.set_trace()
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
    parser = argparse.ArgumentParser(description='Pretraining background inpainting')
    parser.add_argument(
        '--places2filelist',
        help='location of places2 filedict',
        default='./places365_standard/train.txt',
        type=str
    )
    parser.add_argument(
        '--places2_basepath',
        help='location of places2 filedict',
        default='./places365_standard/',
        type=str
    )
    return parser.parse_args()

def main(args):

    ds = Places2DatasetCityScapesMasks(args.places2filelist, args.places2_basepath)
    for idx in range(len(ds)):
        print(idx)
        data=ds[idx]
        # im_texture=utils.texture_from_images_and_iuv(data['im'].unsqueeze(0),data['im_iuv'].unsqueeze(0))
        # utils.plot_texture_map(im_texture.squeeze(0))

if __name__ == '__main__':
    args = parse_args()
    main(args)