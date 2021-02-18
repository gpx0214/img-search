#!/usr/bin/env Python
# -*- coding:UTF-8 -*-

#tested on Python 3.5.2 (v3.5.2:4def2a2901a5, Jun 25 2016, 22:18:55) [MSC v.1900 64 bit (AMD64)] on win32

#Downloading data from:
#https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5 553,467,096Bytes
#https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
#put file on the path next:
#~/.keras/models/ on Linux
#%USERPROFILE%\.keras\models on Windows

import os
import random
import pickle  #import cPickle as pickle #python 2
import numpy as np
import matplotlib.pyplot
from matplotlib.pyplot import imshow
import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model
from sklearn.decomposition import PCA
from scipy.spatial import distance
from tqdm import tqdm
import time

################################################################

def get_image(path):
    img = image.load_img(path, target_size=model.input_shape[1:3])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x

def get_closest_images(feat0, num_results=100):
    distances = [ distance.euclidean(feat0[0], feat) for feat in features ]
    idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[0:num_results+0]
    return idx_closest

#slowly
def get_closest_images_csv(feat0, num_results=100):
    distances = [ distance.euclidean(feat0[0], np.loadtxt(open(image_path+'.csv',"rb"),delimiter=",",skiprows=0) ) for image_path in images ]
    idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[0:num_results+0]
    return idx_closest

def get_concatenated_images(indexes, thumb_height):
    thumbs = []
    for idx in indexes:
        img = image.load_img(images[idx])
        img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
        thumbs.append(img)
    concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
    return concat_image

def pred_idx2htm(predictions,idx_closest,img_width=250,filename=str(int(time.time()))+'.htm'):
    print(filename)
    fimg=open(filename,"w",encoding='utf-8') #
    fimg.write('<!DOCTYPE html> \n')
    fimg.write('<html xmlns="http://www.w3.org/1999/xhtml"  lang="zh-Hans"> \n')
    fimg.write('<head>\n')
    fimg.write('</head>\n')
    fimg.write('<body>\n')
    for pred in decode_predictions(predictions)[0]:
        fimg.write("predicted %s with probability %0.9f<br />  \n" % (pred[1], pred[2])) #%0.3f
    for idx in idx_closest:
        fimg.write("<img width=\"%d\" src=\' %s \'>  \n" % (img_width,(images[idx]).replace("\\", "/")) )
    fimg.write('</body>\n')
    fimg.write('</html>\n')
    fimg.close()

def idx2htm(idx_closest,img_width=250,filename=str(int(time.time()))+'.htm'):
    print(filename)
    fimg=open(filename,"w",encoding='utf-8') #
    fimg.write('<!DOCTYPE html> \n')
    fimg.write('<html xmlns="http://www.w3.org/1999/xhtml"  lang="zh-Hans"> \n')
    fimg.write('<head>\n')
    fimg.write('</head>\n')
    fimg.write('<body>\n')
    for idx in idx_closest:
        fimg.write("<img width=\"%d\" src=\' %s \'>  \n" % (img_width,(images[idx]).replace("\\", "/")) )
    fimg.write('</body>\n')
    fimg.write('</html>\n')
    fimg.close()

################################

model = keras.applications.VGG16(weights='imagenet', include_top=True)
model.summary()

################################

feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)
feat_extractor.summary()

################################################################

images_path = 'D:/DCIM' #'D://20170105'
#max_num_images = 500 #10000

images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(images_path) for f in filenames if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']]
'''
if max_num_images < len(images):
    images = [images[i] for i in sorted(random.sample(range(len(images)), max_num_images))] #xrange(len(images)), max_num_images))] #python2
'''
print("keeping %d images to analyze" % len(images))

################################################################

#1.25s/items
for image_path in tqdm(images):
    img, x = get_image(image_path);
    feat = feat_extractor.predict(x)[0]
    #features.append(feat)
    np.savetxt(image_path+'.csv', feat, delimiter = ',') #np

################################################################

#35items/s read in vector
features = []
for image_path in tqdm(images):
    feat = np.loadtxt(open(image_path+'.csv',"rb"),delimiter=",",skiprows=0) #np
    features.append(feat)

'''
#降维 4096->300 可选
features = np.array(features)
pca = PCA(n_components=300)
pca.fit(features)
pca_features = pca.transform(features)
'''

img, x = get_image("D:\\IMG_20160124_144219_640_480cut.jpg")
feat0 = feat_extractor.predict(x)

idx_closest = get_closest_images(feat0)

#idx2htm(idx_closest,250,'D:\\'+str(int(time.time()))+'.htm')

predictions = model.predict(x) #语义分类可选

pred_idx2htm(predictions,idx_closest,250,'D:\\'+str(int(time.time()))+'.htm')

'''
#文本可选
for pred in decode_predictions(predictions)[0]:
    print("predicted %s with probability %0.9f<br />" % (pred[1], pred[2])) #%0.3f

for idx in idx_closest:
    print("<img width=\"250\" src=\'"+(images[idx]).replace("\\", "/")+"\'>" )
'''

'''
#绘图可选
matplotlib.pyplot.figure(figsize=(16,4))
matplotlib.pyplot.plot(feat0[0])
matplotlib.pyplot.show()

results_image = get_concatenated_images(idx_closest[0:5], 400)
#print("display the query image")
# display the resulting images
matplotlib.pyplot.figure(figsize = (20,15))
imshow(results_image)
matplotlib.pyplot.title("result images")
matplotlib.pyplot.show()
'''

#pickle.dump([images, pca_features], open('D://1.p', 'wb')) #'D://features_caltech101.p'

'''
#计时
print(time.time())
time0=time.time()
img, x = get_image("D:\\IMG_20160124_144219_640_480cut.jpg")
print(time.time()-time0)
time0=time.time()
feat0 = feat_extractor.predict(x)
print(time.time()-time0)
time0=time.time()
idx_closest = get_closest_images(feat0)
print(time.time()-time0)
time0=time.time()
predictions = model.predict(x)
print(time.time()-time0)
time0=time.time()
pred_idx2htm(predictions,idx_closest,250,'D:\\'+str(int(time.time()))+'.htm')
print(time.time()-time0)

'''
