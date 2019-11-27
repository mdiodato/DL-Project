"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
import pandas as pd
import PIL.Image
from data.image_folder import make_dataset

def save_file_path(opt, makedir=False):
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    if makedir:
        util.mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, 'dataload')
    return file_name


opt = TestOptions().parse()

load_data = pd.read_csv(save_file_path(opt) + '.csv')
#dataloader = data.create_dataloader(opt)

test_images_folder = './results/' + opt.name + '/test_' + opt.which_epoch + '/images/' + opt.real_label + '/synthesized_image/'
test_images = make_dataset(test_images_folder)

res = {}
for test_image in test_images:
    test_im = Image.open(test_image)
    test_im = test_im.convert('RGB')
    rows,cols,colors = test_im.shape
    img_size = rows*cols*colors
    test_im = test_im.reshape(img_size)
    
    res_test = []
    for real_image in load_data['image_paths']:
        real_im = Image.open(real_image)
        real_im = real_im.convert('RGB')
        real_im = real_im.resize((rows, cols), Image.BICUBIC)
        rows,cols,colors = real_im.shape
        img_size = rows*cols*colors
        real_im = real_im.reshape(img_size)
        
        l2 = np.linalg.norm(test_im - real_im)
        
    res[test_image] = res_test

pd.DataFrame([res], index = load_data['image_paths'])

res.to_csv('./results/' + opt.name + '/test_' + opt.which_epoch + '/L2 Comparison.csv')