# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 01:09:20 2019

@author: micha
"""

import pandas as pd
import shutil 
import os

input_dir = 'C:/Documents/Education/Columbia/COMS W4995 - Deep Learning/Project/Data/'
output_dir = 'D:/Deep Learning Project/DL-Project/Project Code/datasets/wikiart/'
input_csv_folder = 'wikiart_csv/'
input_folder = 'wikiart/'
output_train = 'train_img/'
output_val = 'val_img/'

genre_map = pd.read_csv(input_dir + input_csv_folder + 'genre_class.txt', header=None, delimiter=r"\s+")
genre_train = pd.read_csv(input_dir + input_csv_folder + 'genre_train.csv')
genre_val = pd.read_csv(input_dir + input_csv_folder + 'genre_val.csv')

for i in range(len(genre_train)):
    input_file = os.path.join(input_dir + input_folder, genre_train.iloc[i, 0])
    output_file = os.path.join(output_dir + output_train, os.path.basename(genre_train.iloc[i, 0]))
    if os.path.isfile(output_file):
        print(output_file)
    else:
        shutil.copyfile(input_file, output_file) 
    
for i in range(len(genre_val)):
    input_file = os.path.join(input_dir + input_folder, genre_val.iloc[i, 0])
    output_file = os.path.join(output_dir + output_train, os.path.basename(genre_val.iloc[i, 0]))
    if os.path.isfile(output_file):
        print(output_file)
    else:
        shutil.copyfile(input_file, output_file) 