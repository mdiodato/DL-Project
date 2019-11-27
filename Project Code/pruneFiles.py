import pandas as pd
import numpy as np
from PIL import Image
import os


data= pd.read_csv('./datasets/wikiart_all/summary.csv')

new_data = pd.DataFrame()
for i in range(len(data)):
    filename = './datasets/wikiart_all/images/' + data.iloc[i,:]['filename'].replace('\\','/')
    try:
        if os.path.isfile(filename):
            tmp = Image.open(filename)
            tmp = tmp.convert('RGB')
            new_data = new_data.append(data.iloc[i,:])

    except Exception as e:
        print(filename, e)

new_data.to_csv('./datasets/wikiart_all/summary filtered.csv')
