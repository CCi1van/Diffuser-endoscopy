'''
Author: MA yifan
Date: 2021-06-13 12:00:42
LastEditTime: 2021-06-17 11:47:49
LastEditors: MA, yifan
Description: 
FilePath: \MNIST-Dataset\fashion\imageGen.py
The MIT License (MIT) Copyright (c) [2021] Ma yifan
'''


import numpy as np
import struct
import matplotlib.pyplot as plt
from matplotlib.image import imsave
import os

# File Path
train_images_file = 'train-images.idx3-ubyte'
train_labels_file = 'train-labels.idx1-ubyte'
test_images_file = 't10k-images.idx3-ubyte'
test_labels_file = 't10k-labels.idx1-ubyte'

def LoadImages(file):
    '''
    Load raw file
        param: 
            File path :str
        return: 
            Image data :ndarray
                D1:images number
                D2:rows
                D3:columns
    '''

    # Read binary data
    with open(file,'rb') as infile:
        data = infile.read()
    
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, rows, cols = struct.unpack_from(fmt_header,data,offset)
    print ('magic number:{}\nimage number:{}\nsize of image:{}x{}'.format(magic_number, num_images, rows, cols))

    image_size = rows * cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images,rows, cols))
    for i in range(num_images):
        if i == 0:
            print('Loading image...')
        print(i+1)
        images[i] = np.array(struct.unpack_from(fmt_image, data, offset)).reshape((rows, cols))
        offset += struct.calcsize(fmt_image)
    return images


def SaveImage(save_path,images):
    '''
    Save image file
        param: 
            save path :str
            images : ndarray
    '''
    os.makedirs(save_path)
    count = 0
    for img in images:
        count +=1
        ima_name = str(count)+'.png'
        file_path = save_path + ima_name
        imsave(file_path,img)
        if count == 1:
            print('Saving image...')    
        print(count)    
        

#Generate Dataset
img_trn = np.array(LoadImages(train_images_file))
img_test = np.array(LoadImages(test_images_file))

plt.imshow(img_trn[0])
plt.axis('off')
plt.show()

SaveImage('TrainingData/',img_trn)
SaveImage('TestData/',img_test)

