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
        #parser.set_defaults(label_nc=13)
        parser.set_defaults(contain_dontcare_label=False)

        #parser.add_argument('--label_dir', type=str, required=True,
        #                    help='path to the directory that contains label images')
        #parser.add_argument('--image_dir', type=str, required=True,
        #                    help='path to the directory that contains photo images')
        #parser.add_argument('--instance_dir', type=str, default='',
        #                    help='path to the directory that contains instance maps. Leave black if not exists')
        return parser

    def get_paths(self, opt):
        if opt.isTrain:
            image_dir_real = opt.image_dir_real 
            image_dir_guide = opt.image_dir_guide 
            
            summary_file_real = pd.read_csv(image_dir_real+'summary.csv', low_memory=False)
            summary_file_real = summary_file_real[summary_file_real[opt.filter_cat_real].isin(opt.filter_values_real)]
            
            summary_file_guide = pd.read_csv(image_dir_guide+'summary.csv', low_memory=False)
            summary_file_guide = summary_file_guide[summary_file_guide[opt.filter_cat_guide].isin(opt.filter_values_guide)]
            
            image_paths_real = (image_dir_real + 'images/' + summary_file_real['filename'].str.replace('\\','/')).tolist()
            image_paths_guide = (image_dir_guide + 'images/' + summary_file_guide['filename'].str.replace('\\','/')).tolist()
            
            le_real = preprocessing.LabelEncoder()
            le_guide = preprocessing.LabelEncoder()
            
            le_real.fit(summary_file_real[opt.filter_cat_real])
            label_paths_real = le_real.transform(summary_file_real[opt.filter_cat_real])
            
            le_guide.fit(summary_file_guide[opt.filter_cat_guide])
            label_paths_guide = le_guide.transform(summary_file_guide[opt.filter_cat_guide])
        
        tmpImg = []
        tmpLab = []
        tmpSty = []
        tmpLabReal = []
        tmpLabGuide= []
        if opt.test_load:
            for i in range(min(len(image_paths_real), len(image_paths_guide), opt.max_dataset_size)):
                try:
                    if os.path.isfile(image_paths_real[i]) and os.path.isfile(image_paths_guide[i]):
                        tmp = Image.open(image_paths_real[i])
                        tmp = tmp.convert('RGB')
                        tmp = Image.open(image_paths_guide[i])
                        tmp = tmp.convert('RGB')
                        tmpImg.append(image_paths_real[i])
                        tmpLab.append(label_paths_real[i])
                        tmpSty.append(image_paths_guide[i])
                        tmpLabReal.append(label_paths_real[i])
                        tmpLabGuide.append(label_paths_guide[i])
                    else:
                        print("Missing files:", image_paths_real[i], image_paths_guide[i])
                except OSError as e:
                    print("OS Error: " + str(e), "File: " + image_paths_real[i], "File: " + image_paths_guide[i])
        else:
            tmpImg = make_dataset(image_dir, recursive=False, read_cache=True)
            tmpLab = tmpImg[:]
            tmpSty = tmpImg[:]
            tmpLabReal = tmpImg[:]
            tmpLabGuide = tmpImg[:]
        image_paths = tmpImg
        label_paths = tmpLab
        style_paths = tmpSty
        label_real = tmpLabReal
        label_guide = tmpLabGuide


        #if len(opt.instance_dir) > 0:
        #    instance_dir = opt.instance_dir
        #    instance_paths = make_dataset(instance_dir, recursive=False, read_cache=True)
        #else:
        #   instance_paths = []
        instance_paths = []
        
        print("Real Image filters:", opt.filter_cat_real, opt.filter_values_real)
        le_name_mapping = dict(zip(le_real.classes_, le_real.transform(le_real.classes_)))
        print(le_name_mapping)

        print("Guide Image filters:", opt.filter_cat_guide, opt.filter_values_guide)
        le_name_mapping = dict(zip(le_guide.classes_, le_guide.transform(le_guide.classes_)))
        print(le_name_mapping)
        
        assert len(label_paths) == len(image_paths), "The #images in %s and %s do not match. Is there something wrong?"
        assert len(style_paths) == len(image_paths), "The #images in %s and %s do not match. Is there something wrong?"

        return label_paths, image_paths, instance_paths, style_paths, label_real, label_guide
