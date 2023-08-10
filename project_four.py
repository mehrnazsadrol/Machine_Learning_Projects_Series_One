import pandas as pd
import matplotlib.pyplot as plt 
from PIL import Image
import cv2
import os

path = './game-of-deep-learning-ship-datasets/train/images/'
df = pd.read_csv('./game-of-deep-learning-ship-datasets/train/train.csv')

df['path'] = path + df['image']
categories = list(df['category'])
category = {1:'Cargo', 2:'Military', 3:'Carrier', 4:'Cruise', 5:'Tankers'}
classes = []
for c in categories: 
    classes.append(category[c])
df['classes'] = classes

test_df = pd.read_csv('./game-of-deep-learning-ship-datasets/test_ApKoW4T.csv')
test_df['path'] = path + test_df['image']


plt.figure(figsize = (15,12))
for idx,image_path in enumerate(df['path']):
    if idx==24:
        break
    plt.subplot(4,8,idx+1)
    img = Image.open(image_path)
    img = img.resize((224,224))
    plt.imshow(img)
    plt.axis('off')
    plt.title(idx)
plt.tight_layout()
plt.show()

plt.figure(figsize = (15,12))
for idx,image_path in enumerate(test_df['path']):
    if idx==24:
        break
    plt.subplot(4,8,idx+1)
    img = Image.open(image_path)
    img = img.resize((224,224))
    plt.imshow(img)
    plt.axis('off')
    plt.title(idx)
plt.tight_layout()
plt.show()

counts = df['classes'].value_counts()
print (counts)
plt.bar(counts.index, counts.values)
name = counts.index
plt.show()


widths, heights = [], []
for i,path in enumerate(df["path"]):
    width, height = Image.open(path).size
    widths.append(width)
    heights.append(height)
    
df["width"] = widths
df["height"] = heights

print (df['width'].describe())
print (df['height'].describe())


r,g,b = [],[],[]

for i,row in df.iterrows(): 
    my_image = cv2.imread(row['path'])
    rgb = my_image.mean(axis=(0,1))
    r.append(rgb[2])
    g.append(rgb[1])
    b.append(rgb[0])

df['red'] = r
df['green'] = g
df['blue'] = b


print (df[['red','green','blue','image']].head())