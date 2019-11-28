"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict
import numpy as np
from options.test_options import TestOptions
import pandas as pd
from PIL import Image
import math
from data.image_folder import make_dataset
from collections import defaultdict

def save_file_path(opt, makedir=False):
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    if makedir:
        util.mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, 'dataload')
    return file_name


opt = TestOptions().parse()

load_data = pd.read_csv(save_file_path(opt) + '.csv')
#dataloader = data.create_dataloader(opt)
print('Loaded information data file')

test_images_folder = './results/' + opt.name + '/test_' + str(opt.which_epoch) + '/images/'
test_images = make_dataset(test_images_folder, recursive=True)
test_images_all = [x for x in test_images if 'synthesized_image' in x]
print('Loaded test images')

res = defaultdict(list)
batch_count = 0
for j in range(0, len(test_images_all), opt.batchSize):
    print("Working on batch coount", batch_count+1, ' of ', len(range(0, len(test_images_all), opt.batchSize)))
    test_arrs = []
    test_images = test_images_all[j:j+opt.batchSize]
    for test_image in test_images:
        #print('Working on ', test_image)
        test_im = Image.open(test_image)
        test_im = test_im.convert('RGB')
        test_im = np.array(test_im)
        rows,cols,colors = test_im.shape
        img_size = rows*cols*colors
        test_im = test_im.ravel()
        test_arrs.append(test_im)
        
    count = 0
    for real_image in load_data['image_paths']:
        if count % math.floor(len(load_data['image_paths'])/10) == 0:
            print('Processed ', count, ' Images of ', len(load_data['image_paths']))
        real_im = Image.open(real_image)
        real_im = real_im.convert('RGB')
        real_im = real_im.resize((rows, cols), Image.BICUBIC)
        real_im = np.array(real_im)
        rows,cols,colors = real_im.shape
        img_size = rows*cols*colors
        real_im = real_im.reshape(img_size)
        count += 1

        for i in range(len(test_images)):
            test_im = test_arrs[i]
            test_image = test_images[i]
            l2 = np.linalg.norm(test_im - real_im)
            
            res[test_image].append(l2)
    batch_count += 1

print('Saving results')
res = pd.DataFrame.from_dict(res,orient='index').transpose() 
res.index = list(load_data['image_paths'])

res.to_csv('./results/' + opt.name + '/test_' + opt.which_epoch + '/L2 Comparison.csv')

with pd.ExcelWriter('./results/' + opt.name + '/test_' + opt.which_epoch + '/L2 Comparison.xlsx') as writer:  # doctest: +SKIP
    res.to_excel(writer, sheet_name='L2 Results')
    load_data.to_excel(writer, sheet_name='Dataloader')
