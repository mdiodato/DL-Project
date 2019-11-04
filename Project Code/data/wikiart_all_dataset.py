"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
import pandas as pd
from sklearn import preprocessing
import os
from PIL import Image


class wikiartalldataset(Pix2pixDataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir, --instance_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        load_size = 286 if is_train else 256
        parser.set_defaults(load_size=load_size)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=13)
        parser.set_defaults(contain_dontcare_label=False)

        #parser.add_argument('--label_dir', type=str, required=True,
        #                    help='path to the directory that contains label images')
        #parser.add_argument('--image_dir', type=str, required=True,
        #                    help='path to the directory that contains photo images')
        #parser.add_argument('--instance_dir', type=str, default='',
        #                    help='path to the directory that contains instance maps. Leave black if not exists')
        return parser

    def get_paths(self, opt):
        label_dir = opt.label_dir
        sumamry_file = pd.read_csv(label_dir+'summary.csv', low_memory=False)
        summary_file = sumamry_file[sumamry_file[opt.filter_cat].isin(opt.filter_values)]
        le = preprocessing.LabelEncoder()
        le.fit(summary_file[opt.filter_cat])
        label_paths = le.transform(summary_file[opt.filter_cat])

        image_dir = opt.image_dir 
        image_paths = (image_dir + 'images/' + summary_file['filename'].str.replace('\\','/')).tolist()
        
        tmpImg = []
        tmpLab = []
        tmpSty = []
        print(image_dir)
        if image_dir == "./datasets/wikiart_all/":
            for i in range(len(image_paths)):
                try:
                    if os.path.isfile(image_paths[i]):
                        tmp = Image.open(image_paths[i])
                        tmp = tmp.convert('RGB')
                        tmpImg.append(image_paths[i])
                        tmpLab.append(label_paths[i])
                    else:
                        print("Missing file:" + image_paths[i])
                except OSError as e:
                    print("OS Error: " + str(e), "File: " + image_paths[i])
            tmpSty = tmpImg[len(tmpImg)//2:]
            tmpImg = tmpImg[:len(tmpImg)//2]
            tmpLab = tmpLab[:len(tmpLab)//2]
            if len(tmpSty) != len(tmpImg):
                print("Style size is uneven:", len(tmpSty), len(tmpImg))
                if len(tmpSty) > len(tmpImg):
                    tmpSty = tmpSty[:-1]
                else:
                    tmpImg = tmpImg[:-1]
        else:
            tmpImg = make_dataset(image_dir, recursive=False, read_cache=True)
            tmpLab = tmpImg[:]
            tmpSty = tmpImg[:]
        image_paths = tmpImg
        label_paths = tmpLab
        style_paths = tmpSty


        #if len(opt.instance_dir) > 0:
        #    instance_dir = opt.instance_dir
        #    instance_paths = make_dataset(instance_dir, recursive=False, read_cache=True)
        #else:
        #   instance_paths = []
        instance_paths = []

        assert len(label_paths) == len(image_paths), "The #images in %s and %s do not match. Is there something wrong?"
        #print(len(style_paths), len(image_paths))
        assert len(style_paths) == len(image_paths), "The #images in %s and %s do not match. Is there something wrong?"

        return label_paths, image_paths, instance_paths, style_paths
