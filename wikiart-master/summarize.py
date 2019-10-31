# -*- coding: utf-8 -*-

import pandas as pd
import json

import os

directory = './wikiart-saved/meta/'
image_directory = './wikiart-saved/images/'
jsons = []
for filename in os.listdir(directory):
    if filename.endswith(".json"): 
        path = os.path.join(directory, filename)
        jsons.append(pd.read_json(path))

jsons = pd.concat(jsons, sort=False)

photos = {f.replace('.jpg',''):os.path.join(path, f).replace(image_directory,'') for path, subdirs, files in os.walk(image_directory) for f in files}

jsons['contentId'] = jsons['contentId'].astype(str)
jsons.drop('description', inplace=True, axis=1)
details = jsons[jsons['contentId'].isin(list(photos.keys()))]
not_details = jsons[~jsons['contentId'].isin(photos)]

details.dropna(axis=1, how='all')

details['filename'] = details['contentId'].map(photos)

details.to_csv('./wikiart-saved/summary.csv', index=False)