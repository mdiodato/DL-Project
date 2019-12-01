import torch
from models.networks.architecture import VGG19StyleAndContent
from collections import OrderedDict
import numpy as np
from options.test_options import TestOptions
import pandas as pd
from PIL import Image
import math
from data.image_folder import make_dataset
from collections import defaultdict
import os
import data
from data.base_dataset import BaseDataset, get_params, get_transform

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Use VGG19 to extract style and content features from the intermediate layers.
vgg = VGG19StyleAndContent().to(device)
style_weights = [0.1, 0.2, 0.4, 0.8, 1.6]


def get_gram_matrix(feature):
    """
    Compute the gram matrix by converting to 2D tensor and doing dot product
    feature: (batch, channel, height, width)
    """
    b, c, h, w = feature.size()
    feature = feature.view(b*c, h*w)
    gram = torch.mm(feature, feature.t())
    return gram


def get_losses(real_img, guide_img, fake_img):
    """
    Calculate the content loss between guide image and fake image
    and the style loss between real image and fake image
    (parameters from https://arxiv.org/pdf/1904.11617.pdf)
    """
    # Get features from VGG19
    _, guide_content = vgg(guide_img)
    real_style, _ = vgg(real_img)
    fake_style, fake_content = vgg(fake_img)

    # Compute content loss
    content_loss = torch.mean((guide_content - fake_content) ** 2)

    # Compute style loss
    style_loss = 0
    for i in range(len(real_style)):
        b, c, h, w = real_style[i].shape
        real_gram = get_gram_matrix(real_style[i])
        fake_gram = get_gram_matrix(fake_style[i])
        layer_style_loss = style_weights[i] * torch.mean((real_gram - fake_gram) ** 2)
        style_loss += layer_style_loss / (c * h * w)

    return style_loss, content_loss

def save_file_path(opt, makedir=False):
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    if makedir:
        util.mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, 'dataload')
    return file_name

def open_image(path, opt):
    image = Image.open(path)
    params = get_params(opt, image.size)
    image = image.convert('RGB')
    transform_image = get_transform(opt, params)
    image_tensor = transform_image(image)
    return image_tensor

opt = TestOptions().parse()
opt.batchSize = 1

load_data = pd.read_csv(save_file_path(opt) + '.csv')
dataloader = data.create_dataloader(opt)
print('Loaded information data file')

test_images_folder = './results/' + opt.name + '/test_' + str(opt.which_epoch) + '/'
test_images = make_dataset(test_images_folder, recursive=True)
test_images_all = [x for x in test_images if 'synthesized_image' in x]
print('Loaded test images')

res = defaultdict(list)
for i, data_i in enumerate(dataloader):
    guide_id = os.path.split(data_i['guide_path'][0])[1].split('.')[0]
    real_id = os.path.split(data_i['path'][0])[1].split('.')[0]
    for test_image in test_images_all:    
        if test_image.split('/')[-3] == real_id and test_image.split('/')[-1].split('.')[0] == guide_id:
            fake_image_path = test_image
            print("Working on", fake_image_path)
            fake_image = open_image(fake_image_path, opt) 

            real_image = data_i['style']
            guide_image = data_i['image']

            real_image = real_image.to(device)
            guide_image = guide_image.to(device)
            fake_image = fake_image.unsqueeze(0).to(device)
            #print(real_image.size(), guide_image.size(), fake_image.size())

            style_loss, content_loss = get_losses(real_image, guide_image, fake_image)
            res[fake_image_path] = [style_loss.item(), content_loss.item()]
            res[fake_image_path].append(data_i['path'])
            res[fake_image_path].append(data_i['guide_path'])
            res[fake_image_path].append(fake_image_path)

print('Saving results')
res = pd.DataFrame.from_dict(res,orient='index')
res.columns = ['Style Loss', 'Content Loss', 'Real Image Path', 'Guide Image Path', 'Fake Image Path']

res.to_csv('./results/' + opt.name + '/test_' + opt.which_epoch + '/Style Comparison.csv')

with pd.ExcelWriter('./results/' + opt.name + '/test_' + opt.which_epoch + '/Style Comparison.xlsx') as writer:  # doctest: +SKIP
    res.to_excel(writer, sheet_name='Style Results')
    load_data.to_excel(writer, sheet_name='Dataloader')
