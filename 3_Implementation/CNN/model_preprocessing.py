src_folder = "../data/Vegetation/training"
train_folder = "../data/Vegetation/ConvTrain"
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


fig = plt.figure(figsize=(12, 16))

dir_num = 0
for root, folders, filenames in os.walk(src_folder):
    for folder in folders:
        file = os.listdir(os.path.join(root,folder))[0]
        imgFile = os.path.join(root,folder, file)
        img = Image.open(imgFile)
        a=fig.add_subplot(4,np.ceil(len(folders)/4),dir_num + 1)
        imgplot = plt.imshow(img)
        a.set_title(folder)
        dir_num = dir_num + 1

def resize_image(src_image, size=(128,128), bg_color="white"): 
    from PIL import Image, ImageOps 
    
    src_image.thumbnail(size, Image.ANTIALIAS)
    new_image = Image.new("RGB", size, bg_color)
    new_image.paste(src_image, (int((size[0] - src_image.size[0]) / 2), int((size[1] - src_image.size[1]) / 2)))
    return new_image
size = (128,128)
if os.path.exists(train_folder):
    shutil.rmtree(train_folder)
for root, folders, files in os.walk(src_folder):
    for sub_folder in folders:
        print('processing folder ' + sub_folder)
        saveFolder = os.path.join(train_folder,sub_folder)
        if not os.path.exists(saveFolder):
            os.makedirs(saveFolder)
        file_names = os.listdir(os.path.join(root,sub_folder))
        for file_name in file_names:
            file_path = os.path.join(root,sub_folder, file_name)
            print("reading " + file_path)
            image = Image.open(file_path)
            resized_image = resize_image(image, size)
            saveAs = os.path.join(saveFolder, file_name)
            print("writing " + saveAs)
            resized_image.save(saveAs)
fig = plt.figure(figsize=(12,12))

image_num = 0
for root, folders, filenames in os.walk(src_folder):
    for folder in folders:
        file = os.listdir(os.path.join(root,folder))[0]
        src_file = os.path.join(src_folder,folder, file)
        src_image = Image.open(src_file)
        image_num += 1
        a=fig.add_subplot(len(folders), 2, image_num)
        imgplot = plt.imshow(src_image)
        a.set_title(folder)
        resized_file = os.path.join(train_folder,folder, file)
        resized_image = Image.open(resized_file)
        image_num += 1
        b=fig.add_subplot(len(folders), 2, image_num)
        imgplot = plt.imshow(resized_image)
        b.set_title('resized ' + folder)
src_folder = "../data/Vegetation/training"
train_folder = "../data/Vegetation/ConvTrain"
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

fig = plt.figure(figsize=(12, 16))

dir_num = 0
for root, folders, filenames in os.walk(src_folder):
    for folder in folders:
        file = os.listdir(os.path.join(root,folder))[0]
        imgFile = os.path.join(root,folder, file)
        img = Image.open(imgFile)
        a=fig.add_subplot(4,np.ceil(len(folders)/4),dir_num + 1)
        imgplot = plt.imshow(img)
        a.set_title(folder)
        dir_num = dir_num + 1
def resize_image(src_image, size=(128,128), bg_color="white"): 
    from PIL import Image, ImageOps 

    src_image.thumbnail(size, Image.ANTIALIAS)
    new_image = Image.new("RGB", size, bg_color)
    new_image.paste(src_image, (int((size[0] - src_image.size[0]) / 2), int((size[1] - src_image.size[1]) / 2)))
    return new_image
size = (128,128)
if os.path.exists(train_folder):
    shutil.rmtree(train_folder)
for root, folders, files in os.walk(src_folder):
    for sub_folder in folders:
        print('processing folder ' + sub_folder)
        saveFolder = os.path.join(train_folder,sub_folder)
        if not os.path.exists(saveFolder):
            os.makedirs(saveFolder)
        file_names = os.listdir(os.path.join(root,sub_folder))
        for file_name in file_names:
            file_path = os.path.join(root,sub_folder, file_name)
            print("reading " + file_path)
            image = Image.open(file_path)
            resized_image = resize_image(image, size)
            saveAs = os.path.join(saveFolder, file_name)
            print("writing " + saveAs)
            resized_image.save(saveAs)
            
fig = plt.figure(figsize=(12,12))
image_num = 0
for root, folders, filenames in os.walk(src_folder):
    for folder in folders:
        file = os.listdir(os.path.join(root,folder))[0]
        src_file = os.path.join(src_folder,folder, file)
        src_image = Image.open(src_file)
        image_num += 1
        a=fig.add_subplot(len(folders), 2, image_num)
        imgplot = plt.imshow(src_image)
        a.set_title(folder)
        resized_file = os.path.join(train_folder,folder, file)
        resized_image = Image.open(resized_file)
        image_num += 1
        b=fig.add_subplot(len(folders), 2, image_num)
        imgplot = plt.imshow(resized_image)
        b.set_title('resized ' + folder)
