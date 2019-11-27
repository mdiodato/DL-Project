from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import pandas as pd

#Code is modified from https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb

dataTypes = ['train2017', 'val2017']
df_all = []
for dataType in dataTypes:
    dataDir='..'
    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

    coco=COCO(annFile)

    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    nms=[cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))

    super_nms = list(set([cat['supercategory'] for cat in cats]))
    print('COCO supercategories: \n{}'.format(' '.join(super_nms)))

    image_cat = {}
    images = []
    for i in range(len(nms)):
        # get all images containing given categories, select one at random
        catIds = coco.getCatIds(catNms=nms[i]);
        for j in catIds:
            imgIds = coco.getImgIds(catIds=[j]);
            image_cat[nms[i]] = imgIds
            for Id in imgIds:
                tmp=[]
                tmp.append(Id)
                tmp.append(nms[i])
                tmp.append('')
                images.append(tmp)

    image_super = {}
    for i in range(len(super_nms)):
        # get all images containing given categories, select one at random
        catIds = coco.getCatIds(supNms=super_nms[i]);
        #print(super_nms[i], catIds)
        for j in catIds:
            imgIds = coco.getImgIds(catIds=[j]);
            image_super[super_nms[i]] = imgIds
            #print(super_nms[i], len(imgIds))
            for Id in imgIds:
                tmp = []
                tmp.append(Id)
                tmp.append('')
                tmp.append(super_nms[i])
                images.append(tmp)

    df = pd.DataFrame(images, columns = ['Image ID', 'Category', 'Super Category'])
    df['filename'] = df['Image ID'].astype(str).str.zfill(12) + '.jpg'
    df_all.append(df)

df = pd.concat(df_all)
df.to_csv('../../Project Code/datasets/coco/summary.csv')

    #images = list(set(x for l in list(image_cat.values()) for x in l) | set(x for l in list(image_super.values()) for x in l))


