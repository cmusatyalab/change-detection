'''
Create and save list of all file paths in the UCF dataset
To be used in the UCF dataset loader
'''

import argparse
import os
from pdb import set_trace as st
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description='Extract file lists from preprocessed paths')
    parser.add_argument(
        '--train',
        help='location of VL-CMU-CD train folder',
        default='./train',
        type=str
    )
    parser.add_argument(
        '--test',
        help='location of VL-CMU-CD test folder',
        default='./test',
        type=str
    )
    parser.add_argument(
        '--train_list',
        help='location of VL-CMU-CD train dict',
        default='./train/train.txt',
        type=str
    )
    parser.add_argument(
        '--test_list',
        help='location of VL-CMU-CD test dict',
        default='./test/test.txt',
        type=str
    )
    return parser.parse_args()

def main(args):
    train_dict = {}
    train_dict['gt'] = []
    train_dict['im1'] = []
    train_dict['im2'] = []

    for i, _ in enumerate(os.listdir(args.train + '/gt')):
        train_dict['gt'].append(args.train + '/gt/gt' + str(i) + '.png')
        train_dict['im1'].append(args.train + '/im1/1_im' + str(i) + '.png')
        train_dict['im2'].append(args.train + '/im2/2_im' + str(i) + '.png')

    test_dict = {}
    test_dict['gt'] = []
    test_dict['im1'] = []
    test_dict['im2'] = []

    for i, _ in enumerate(os.listdir(args.test + '/gt')):
        test_dict['gt'].append(args.test + '/gt/gt' + str(i) + '.png')
        test_dict['im1'].append(args.test + '/im1/1_im' + str(i) + '.png')
        test_dict['im2'].append(args.test + '/im2/2_im' + str(i) + '.png')

    with open(args.train_list, "wb") as fp:
        pickle.dump(train_dict, fp)
    
    with open(args.test_list, "wb") as fp:
        pickle.dump(test_dict, fp)

if __name__ == '__main__':
    args = parse_args()
    main(args)