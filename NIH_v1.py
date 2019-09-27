# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:51:56 2019

@author: mohammad.yusuf
"""

# Code from :  https://www.kaggle.com/aakashnain/beating-everything-with-depthwise-convolution

# --------------------------
# This code is for converted CT slices to PNG and then using the pneumonia model to classify
# CT slices were converted by ImageMagick or xConvert using the batch command
#            for /r %i in (*) do magick %i %i.png
# Code is from Imran_pneumonia_v3

# also helpful code in https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/
# https://www.analyticsvidhya.com/blog/2019/04/build-first-multi-label-image-classification-model-python/

#--------------------------

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import glob
import h5py
import shutil
import imgaug as aug
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import imgaug.augmenters as iaa
import sklearn
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
from keras.models import Sequential, Model

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_confusion_matrix
import cv2
from keras import backend as K
color = sns.color_palette()
%matplotlib inline

#from keras.applications import InceptionV3
#from keras.applications.inception_v3 import preprocess_input

# --------- From DICOM proc-----------
import os
import numpy as np
import pandas as pd
import argparse
#import gdcm
from PIL import Image

import pydicom as dicom
import scipy.ndimage

%matplotlib inline
import sys
from numpy import *
from scipy import stats
import glob
import datetime
import time
import math
import os.path
from importlib import reload
import matplotlib.pyplot as plt
from IPython.display import display
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from multiprocessing import Pool
from skimage import measure, morphology,segmentation
import scipy.ndimage as ndimage
import tensorflow as tf
import pydicom.uid
import matplotlib.gridspec as gridspec
from glob import glob
from keras.preprocessing import image
from keras.callbacks import LambdaCallback
from keras.regularizers import l2
#------------------------------------

#------Dataset defs-----------
#patient_dir = 'D:/Datasets/Dicom_Test/smallsetTrain/'
train_dir = 'F:/Dataset2/NIHChest/images/images'
valid_dir='F:/Dataset2/NIHChest/images/valid'
test_dir= 'F:/Dataset2/NIHChest/images/test'
#resample_dir = args.resampled
##resample_dir = 'D:/Datasets/Dicom_Test/smallsetTrain/'
#resample_dir = 'D:/Datasets/TCIAsmall/Train/'
##resample_dir_train = 'D:/Datasets/Dicom_Test/smallsetTrain/'
#resample_dir_train = 'D:/Datasets/TCIAsmall/Train/'
#patients_csv = args.csv
train_csv = 'F:/Dataset2/NIHChest/Data_Entry_2017.csv'
valid_csv    = 'F:/Dataset2/NIHChest/Data_Entry_2017_valid.csv'
test_csv    = 'F:/Dataset2/NIHChest/Data_Entry_2017_test.csv'
#resample_dir_valid = 'D:/Datasets/TCIAsmall/Valid/'
#aug_dir = 'D:/Datasets/TCIAsmall/AugTrain/'
#valid_aug_dir = 'D:/Datasets/TCIAsmall/AugValid/'

# Disable multi-threading in tensorflow ops
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# Define a tensorflow session with above session configs
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

# Set the session in keras
K.set_session(sess)

# Make the augmentation sequence deterministic
aug.seed(111)

#from progressbar import ProgressBar
#pbar = ProgressBar()
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--patients', '-p', type=str, help="Path to patient folders directory")
parser.add_argument('--resampled', '-r', type=str, help="Path to directory with preprocessed images")
parser.add_argument('--csv', '-c', type=str, help="Path to a csv file with patient id's and their diagnosis")
parser.add_argument('--downsize', '-d', nargs='?', default=50, type=int, help="Value to size downsize the patient images to")
parser.add_argument('--tslices', '-ts', nargs='?', default=20, type=int, help="How many slices the image should be resampled to")
parser.add_argument('--epochs', '-e', nargs='?', default=10, type=int, help="How many training epochs to use")
parser.add_argument('--saver', '-s', nargs='?', type=str, help="Path to save a tensor training model save file")
args = parser.parse_args()

slice_count = args.tslices
downsize_shape = args.downsize
#epochs = args.epochs
epochs = 2
saver = None
augmnt=1

patientPathArray = []       # Holds the paths to the patient image data directories
processPatientArray = []    # Holds the paths to the unprocessed patient image directories

validPathArray = []       # Holds the paths to the valid image data directories
processValidArray = []    # Holds the paths to the unprocessed valid image directories

testPathArray = []       # Holds the paths to the test image data directories
processTestArray = []    # Holds the paths to the unprocessed test image directories



# ***********************************************************
# TRAIN data 
# Path to train directory (Fancy pathlib...no more os.path!!)
traindata  = pd.read_csv(train_csv)

#drop unused columns
traindata = traindata[['Image Index','Finding Labels','Patient Gender']]
traindata.shape

images = glob(os.path.join(train_dir, "*.png"))
images[0:10]
X,y = proc_train_images()

train_labels=y

# One hot encode the training data
e=to_categorical(y)
#traindata.columns
p=np.array(X)
p.shape


x_train=p
y_train=e

x_train.shape
y_train.shape


RareClass = ["Edema", "Emphysema", "Fibrosis", "Pneumonia", "Pleural_Thickening", "Cardiomegaly","Hernia"]
disease=["No Finding","Consolidation","Infiltration","Pneumothorax","Edema","Emphysema","Fibrosis","Effusion","Pneumonia","Pleural_Thickening","Cardiomegaly","Mass","Hernia","Atelectasis"]

#s='Effusion|Mass|Nodule|Pneumothorax'
#z=(s.find("|"))
#print(z)
#print(s[:(s.find("|"))])

#m=33
#plt.imshow(p[m])
#traindata['Image Index'][m]
#traindata['Finding Labels'][m]
#img= 'F:/Dataset2/NIHChest/images/images\\00026451_004.png'

def proc_train_images():
    """
    Returns two arrays: 
        x is an array of resized images
        y is an array of labels
    """
    timage=[]
    NoFinding = "No Finding" #0
    Consolidation="Consolidation" #1
    Infiltration="Infiltration" #2
    Pneumothorax="Pneumothorax" #3
    Edema="Edema" # 7
    Emphysema="Emphysema" #7
    Fibrosis="Fibrosis" #7
    Effusion="Effusion" #4
    Pneumonia="Pneumonia" #7
    Pleural_Thickening="Pleural_Thickening" #7
    Cardiomegaly="Cardiomegaly" #7
    Mass="Mass" #5
    Hernia="Hernia" #7
    Atelectasis="Atelectasis"  #6 
    RareClass = ["Edema", "Emphysema", "Fibrosis", "Pneumonia", "Pleural_Thickening", "Cardiomegaly","Hernia"]
    x = [] # images as arrays
    y = [] # labels
    WIDTH = 128
    HEIGHT = 128
    for img in tqdm(images):
        base = os.path.basename(img)
        # Read and resize image
        full_size_image = cv2.imread(img)
        finding = traindata["Finding Labels"][traindata["Image Index"] == base].values[0]
        symbol = "|"
        #print(base, '     ', finding,'          ', img)
        # Process image
        timg = image.load_img(img,target_size=(224,224,3))
        timg = image.img_to_array(timg)
        timg = timg/255
        timage.append(timg)
        # Process labels
        if symbol in finding:
            #continue
            finding = finding[:(finding.find("|"))]
        if finding == 'Nodule':
            finding='Mass'
        #else:
        if NoFinding in finding:
            finding = 0
            y.append(finding)
            x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC)) 
            #print(finding)
        elif Consolidation in finding:
            finding = 1
            y.append(finding)
            x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        elif Infiltration in finding:
            finding = 2
            y.append(finding)
            x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        elif Pneumothorax in finding:
            finding = 3
            y.append(finding)
            x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        elif Edema in finding:
            finding = 4
            y.append(finding)
            x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        elif Emphysema in finding:
            finding = 5
            y.append(finding)
            x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        elif Fibrosis in finding:
            finding = 6
            y.append(finding)
            x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        elif Effusion in finding:
            finding = 7
            y.append(finding)
            x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        elif Pneumonia in finding:
            finding = 8
            y.append(finding)
            x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        elif Pleural_Thickening in finding:
            finding = 9
            y.append(finding)
            x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        elif Cardiomegaly in finding:
            finding = 10
            y.append(finding)
            x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        #elif NoduleMass in finding:
        elif Mass in finding:
            finding = 11
            y.append(finding)
            x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        elif Hernia in finding:
            finding = 12
            y.append(finding)
            x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        elif Atelectasis in finding:
            finding = 13
            y.append(finding)
            x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
            #print(finding)                
        else:
            print(base)
            continue
        #print(y)
    #return x,y
    return timage,y




# ---------------------------------------------------------
# Plot data distribution
# taken from Project ... Chest_1_Working.py
    

labels = pd.read_csv(train_csv)
labels.head(10)
labels = labels[['Image Index','Finding Labels','Follow-up #','Patient ID','Patient Age','Patient Gender']]
#create new columns for each decease
pathology_list = ['Cardiomegaly','Emphysema','Effusion','Hernia','Nodule','Pneumothorax','Atelectasis','Pleural_Thickening','Mass','Edema','Consolidation','Infiltration','Fibrosis','Pneumonia']
for pathology in pathology_list :
    labels[pathology] = labels['Finding Labels'].apply(lambda x: 1 if pathology in x else 0)
#remove Y after age
labels['Age']=labels['Patient Age'].apply(lambda x: x[:-1]).astype(int)




plt.figure(figsize=(15,10))
gs = gridspec.GridSpec(8,1)
ax1 = plt.subplot(gs[:7, :])
ax2 = plt.subplot(gs[7, :])
data1 = pd.melt(labels,
             id_vars=['Patient Gender'],
             value_vars = list(pathology_list),
             var_name = 'Category',
             value_name = 'Count')
data1 = data1.loc[data1.Count>0]
g=sns.countplot(y='Category',hue='Patient Gender',data=data1, ax=ax1, order = data1['Category'].value_counts().index)
ax1.set( ylabel="",xlabel="")
ax1.legend(fontsize=20)
ax1.set_title('X Ray partition',fontsize=18);
labels['Nothing']=labels['Finding Labels'].apply(lambda x: 1 if 'No Finding' in x else 0)
data2 = pd.melt(labels,
             id_vars=['Patient Gender'],
             value_vars = list(['Nothing']),
             var_name = 'Category',
             value_name = 'Count')
data2 = data2.loc[data2.Count>0]
g=sns.countplot(y='Category',hue='Patient Gender',data=data2,ax=ax2)
ax2.set( ylabel="",xlabel="Number of decease")
ax2.legend('')
plt.subplots_adjust(hspace=.5)




f, ax = plt.subplots(sharex=True,figsize=(15, 10))
g=sns.countplot(y='Category',data=data1, ax=ax, order = data1['Category'].value_counts().index,color='b',label="Multiple Pathologies")
sns.set_color_codes("muted")
g=sns.barplot(x='Patient ID',y='Finding Labels',data=df2, ax=ax, color="r",label="Single Pathology")
ax.legend(ncol=2, loc="center right", frameon=True,fontsize=20)
ax.set( ylabel="",xlabel="Number of Patients")
ax.set_title("Comparaison between Single or Multiple Pathologies",fontsize=20)      
sns.despine(left=True)




# -----------------------------------------------
# Balancing the skewed datasets
# taken from Project ... Chest_1_Working.py

#The imbalance in our dataset has resulted in a biased model. I tried to prevent this by modifying 
#the class_weights parameter and using in the model.fit function but apparently that was not enough. 
#Now I will try to compensate for the imbalanced sample size by oversampling or upsampling the minority classes.

df = pd.DataFrame()
df["images"]=X
df["labels"]=y
print(len(df), df.images[0].shape)
print(type(X))

#Describe new numpy arrays
dict_characters = {1: 'Consolidation', 2: 'Infiltration', 
        3: 'Pneumothorax', 4:'Effusion', 5: 'Nodule Mass', 6: 'Atelectasis', 7: "Other Rare Classes"}

lab = df['labels']
dist = lab.value_counts()
sns.countplot(lab)
print(dict_characters)


#It is very import to do upsampling AFTER the train_test_split function otherwise you can end 
#up with values in the testing dataset that are related to the values within the training dataset.
#X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
# Reduce Sample Size for DeBugging
X_train=x_train
Y_train=train_labels
X_test=x_test
Y_test=test_labels


X_train = X_train[0:5000] 
Y_train = Y_train[0:5000]
X_test = X_test[0:2000] 
Y_test = Y_test[0:2000]




# Make Data 1D for compatability upsampling methods
X_trainShape = X_train.shape[1]*X_train.shape[2]*X_train.shape[3]
X_testShape = X_test.shape[1]*X_test.shape[2]*X_test.shape[3]
X_trainFlat = X_train.reshape(X_train.shape[0], X_trainShape)
X_testFlat = X_test.reshape(X_test.shape[0], X_testShape)
print("X_train Shape: ",X_train.shape)
print("X_test Shape: ",X_test.shape)
print("X_trainFlat Shape: ",X_trainFlat.shape)
print("X_testFlat Shape: ",X_testFlat.shape)


# ROS=Random Over Sampler
import imblearn
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(ratio='auto')
X_trainRos, Y_trainRos = ros.fit_sample(X_trainFlat, Y_train)
X_testRos, Y_testRos = ros.fit_sample(X_testFlat, Y_test)

# Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_trainRosHot = to_categorical(Y_trainRos, num_classes = 14)
Y_testRosHot = to_categorical(Y_testRos, num_classes = 14)
print("X_train: ", X_train.shape)
print("X_trainFlat: ", X_trainFlat.shape)
print("X_trainRos Shape: ",X_trainRos.shape)
print("X_testRos Shape: ",X_testRos.shape)
print("Y_trainRos Shape: ",Y_trainRosHot.shape)
print("Y_testRos Shape: ",Y_testRosHot.shape)
print("Y_trainRosHot Shape: ",Y_trainRosHot.shape)
print("Y_testRosHot Shape: ",Y_testRosHot.shape)




for i in range(len(X_trainRos)):
    height, width, channels = 224,224,3
    X_trainRosReshaped = X_trainRos.reshape(len(X_trainRos),height,width,channels)
print("X_trainRos Shape: ",X_trainRos.shape)
print("X_trainRosReshaped Shape: ",X_trainRosReshaped.shape)

for i in range(len(X_testRos)):
    height, width, channels = 224,224,3
    X_testRosReshaped = X_testRos.reshape(len(X_testRos),height,width,channels)
print("X_testRos Shape: ",X_testRos.shape)
print("X_testRosReshaped Shape: ",X_testRosReshaped.shape)


# Convert labels to one-hot as well for model#4 processing
Y_trainReshaped = to_categorical(Y_trainRos)
Y_testReshaped = to_categorical(Y_testRos)
print("X_trainReshaped Shape: ",Y_trainReshaped.shape)
print("X_testReshaped Shape: ",Y_testReshaped.shape)





dfRos = pd.DataFrame()
dfRos["labels"]=Y_trainRos
labRos = dfRos['labels']
distRos = lab.value_counts()
sns.countplot(labRos)
print(dict_characters)



##Now we have a much more even distriution of sample sizes for each of our 7 ailments 
##(plus an 8th category for other/typos). This should help make our model less biased in favor 
##of the majority class (0=No Finding).
#from sklearn.utils import class_weight
#class_weight = class_weight.compute_class_weight('balanced', np.unique(y), y)
#print("Old Class Weights: ",class_weight)
#from sklearn.utils import class_weight
#class_weight = class_weight.compute_class_weight('balanced', np.unique(Y_trainRos), Y_trainRos)
#print("New Class Weights: ",class_weight)



    




# ***********************************************************
#Test data
testdata  = pd.read_csv(test_csv)

#drop unused columns
testdata = testdata[['Image Index','Finding Labels','Patient Gender']]
testdata.shape

images = glob(os.path.join(test_dir, "*.png"))
images[0:10]
testX,testy = proc_test_images()

test_labels=testy

# One hot encode the training data
e=to_categorical(testy)
#traindata.columns
p=np.array(testX)
p.shape


x_test=p
y_test=e

x_test.shape
y_test.shape



#m=33
#plt.imshow(p[m])
#traindata['Image Index'][m]
#traindata['Finding Labels'][m]


def proc_test_images():
    """
    Returns two arrays: 
        x is an array of resized images
        y is an array of labels
    """
    timage=[]
    NoFinding = "No Finding" #0
    Consolidation="Consolidation" #1
    Infiltration="Infiltration" #2
    Pneumothorax="Pneumothorax" #3
    Edema="Edema" # 4
    Emphysema="Emphysema" #5
    Fibrosis="Fibrosis" #6
    Effusion="Effusion" #7
    Pneumonia="Pneumonia" #8
    Pleural_Thickening="Pleural_Thickening" #9
    Cardiomegaly="Cardiomegaly" #10
    Mass="Mass" #11
    Hernia="Hernia" #12
    Atelectasis="Atelectasis"  #13
    RareClass = ["Edema", "Emphysema", "Fibrosis", "Pneumonia", "Pleural_Thickening", "Cardiomegaly","Hernia"]
    x = [] # images as arrays
    y = [] # labels
    WIDTH = 128
    HEIGHT = 128
    for img in tqdm(images):
        base = os.path.basename(img)
        # Read and resize image
        full_size_image = cv2.imread(img)
        finding = testdata["Finding Labels"][testdata["Image Index"] == base].values[0]
        symbol = "|"
        #print(base, '     ', finding,'          ', img)
        # Process image
        timg = image.load_img(img,target_size=(224,224,3))
        timg = image.img_to_array(timg)
        timg = timg/255
        timage.append(timg)
        # Process labels
        if symbol in finding:
            #continue
            finding = finding[:(finding.find("|"))]
        if finding == 'Nodule':
            finding='Mass'
        #else:
        if NoFinding in finding:
            finding = 0
            y.append(finding)
            x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC)) 
            #print(finding)
        elif Consolidation in finding:
            finding = 1
            y.append(finding)
            x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        elif Infiltration in finding:
            finding = 2
            y.append(finding)
            x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        elif Pneumothorax in finding:
            finding = 3
            y.append(finding)
            x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        elif Edema in finding:
            finding = 4
            y.append(finding)
            x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        elif Emphysema in finding:
            finding = 5
            y.append(finding)
            x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        elif Fibrosis in finding:
            finding = 6
            y.append(finding)
            x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        elif Effusion in finding:
            finding = 7
            y.append(finding)
            x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        elif Pneumonia in finding:
            finding = 8
            y.append(finding)
            x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        elif Pleural_Thickening in finding:
            finding = 9
            y.append(finding)
            x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        elif Cardiomegaly in finding:
            finding = 10
            y.append(finding)
            x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        #elif NoduleMass in finding:
        elif Mass in finding:
            finding = 11
            y.append(finding)
            x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        elif Hernia in finding:
            finding = 12
            y.append(finding)
            x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        elif Atelectasis in finding:
            finding = 13
            y.append(finding)
            x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
            #print(finding)                
        else:
            print(base)
            continue
        #print(y)
    #return x,y
    return timage,y




# ###############################################
#  Disease Key
    
#    NoFinding = "No Finding" #0
#    Consolidation="Consolidation" #1
#    Infiltration="Infiltration" #2
#    Pneumothorax="Pneumothorax" #3
#    Edema="Edema" # 4
#    Emphysema="Emphysema" #5
#    Fibrosis="Fibrosis" #6
#    Effusion="Effusion" #7
#    Pneumonia="Pneumonia" #8
#    Pleural_Thickening="Pleural_Thickening" #9
#    Cardiomegaly="Cardiomegaly" #10
#    Mass="Mass" #11
#    Hernia="Hernia" #12
#    Atelectasis="Atelectasis"  #13
    
# ###############################################
    




#--------------------------------------
# Model 1
# This is with unbalanced datasets
    
print(x_train.shape)
print(y_train.shape)

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu',name='Conv2'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25,name='Dropout2'))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu",name='Conv3'))
model.add(MaxPooling2D(pool_size=(2, 2),name='max2'))
model.add(Dropout(0.25,name='Dropout4'))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu',name='Conv4'))
model.add(MaxPooling2D(pool_size=(2, 2),name='max3'))
model.add(Dropout(0.25,name='Dropout5'))
model.add(Flatten())
model.add(Dense(128, activation='relu',name='dense1'))
model.add(Dropout(0.5,name='dropout6'))
model.add(Dense(64, activation='relu',name='dense2'))
model.add(Dropout(0.5,name='dropouut7'))
model.add(Dense(14, activation='sigmoid',name='dense3-Model-1'))
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

d=np.array[disease]

print(y_train[19])

history=model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test), batch_size=64)

# Mass
q='F:/Dataset2/NIHChest/images/valid/00030159_000.png'
#Pneumothorax
q='F:/Dataset2/NIHChest/images/valid/00030323_030.png'
#Atelectasis
q='F:/Dataset2/NIHChest/images/valid/00030300_002.png'

img = image.load_img(q,target_size=(224,224,3))
img = image.img_to_array(img)
img = img/255
#classes = np.array(train.columns[2:])
proba = model.predict(img.reshape(1,224,224,3))
top_3 = np.argsort(proba[0])[:-4:-1]
print(top_3)
for i in range(3):
    print("{}".format(classes[top_3[i]])+" ({:.3})".format(proba[0][top_3[i]]))
plt.imshow(img)



# -------------------------------------------------------
# Model 2
# This is with unbalanced datasets

# Augmentation sequence 
seq = iaa.OneOf([
    iaa.Fliplr(), # horizontal flips
    iaa.Affine(rotate=20), # roatation
    iaa.Multiply((1.2, 1.5))]) #random brightness
    
    
    
#Training data generator
#Here I will define a very simple data generator. You can do more than this if you want but I think at this point,
# this is more than enough I need.
def data_gen(data, batch_size):
    #data=train_data
    #batch_size=1
    # Get total number of samples in the data
    n = len(data)
    steps = n//batch_size
    
    # Define two numpy arrays for containing batch data and labels
    batch_data = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)
    batch_labels = np.zeros((batch_size,2), dtype=np.float32)

    # Get a numpy array of all the indices of the input data
    indices = np.arange(n)
    
    # Initialize a counter
    i =0
    while True:
        np.random.shuffle(indices)
        # Get the next batch 
        count = 0
        next_batch = indices[(i*batch_size):(i+1)*batch_size]
        for j, idx in enumerate(next_batch):
            #print(j,idx)
            #j=1
            #idx=1
            img_name = data.iloc[idx]['image']
            label = data.iloc[idx]['label']
            
            # one hot encoding
            encoded_label = to_categorical(label, num_classes=2)
            # read the image and resize
            img = cv2.imread(str(img_name))
            img = cv2.resize(img, (224,224))
            
            # check if it's grayscale
            if img.shape[2]==1:
                img = np.dstack([img, img, img])
            
            # cv2 reads in BGR mode by default
            orig_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # normalize the image pixels
            orig_img = img.astype(np.float32)/255.
            
            batch_data[count] = orig_img
            #print(batch_data[1])
            batch_labels[count] = encoded_label
            
            # generating more samples of the undersampled class
            #if label==0 and count < batch_size-2:
            if label==1 and count < batch_size-2:
                #print(label,'--augmenting..', end='')
                aug_img1 = seq.augment_image(img)
                aug_img2 = seq.augment_image(img)
                aug_img1 = cv2.cvtColor(aug_img1, cv2.COLOR_BGR2RGB)
                aug_img2 = cv2.cvtColor(aug_img2, cv2.COLOR_BGR2RGB)
                aug_img1 = aug_img1.astype(np.float32)/255.
                aug_img2 = aug_img2.astype(np.float32)/255.

                batch_data[count+1] = aug_img1
                batch_labels[count+1] = encoded_label
                batch_data[count+2] = aug_img2
                batch_labels[count+2] = encoded_label
                count +=2
                #print('1',end='')
                
            else:
                count+=1
            
            if count==batch_size-1:
                break
            
        i+=1
        yield batch_data, batch_labels
            
        if i>=steps:
            i=0


def build_model():
    input_img = Input(shape=(224,224,3), name='ImageInput')
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv1_1')(input_img)
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv1_2')(x)
    x = MaxPooling2D((2,2), name='pool1')(x)
    
    x = SeparableConv2D(128, (3,3), activation='relu', padding='same', name='Conv2_1')(x)
    x = SeparableConv2D(128, (3,3), activation='relu', padding='same', name='Conv2_2')(x)
    x = MaxPooling2D((2,2), name='pool2')(x)
    
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_1')(x)
    x = BatchNormalization(name='bn1')(x)
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_3')(x)
    x = MaxPooling2D((2,2), name='pool3')(x)
    
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_1')(x)
    x = BatchNormalization(name='bn3')(x)
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_2')(x)
    x = BatchNormalization(name='bn4')(x)
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_3')(x)
    x = MaxPooling2D((2,2), name='pool4')(x)
    
    x = Flatten(name='flatten')(x)
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dropout(0.7, name='dropout1')(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dropout(0.5, name='dropout2')(x)
    x = Dense(14, activation='softmax', name='fc3')(x)
    
    model = Model(inputs=input_img, outputs=x)
    return model
model =  build_model()
model.summary()




#log_dir="C:/TensorBoardLog/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") 
log_dir="C:/TensorBoardLog/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)




#We will initialize the weights of first two convolutions with imagenet weights,
# Open the VGG16 weight file
f = h5py.File('D:/chest_xray/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', 'r')

# Select the layers for which you want to set weight.

w,b = f['block1_conv1']['block1_conv1_W_1:0'], f['block1_conv1']['block1_conv1_b_1:0']
model.layers[1].set_weights = [w,b]

w,b = f['block1_conv2']['block1_conv2_W_1:0'], f['block1_conv2']['block1_conv2_b_1:0']
model.layers[2].set_weights = [w,b]

w,b = f['block2_conv1']['block2_conv1_W_1:0'], f['block2_conv1']['block2_conv1_b_1:0']
model.layers[4].set_weights = [w,b]

w,b = f['block2_conv2']['block2_conv2_W_1:0'], f['block2_conv2']['block2_conv2_b_1:0']
model.layers[5].set_weights = [w,b]

f.close()
model.summary()    


# Get a train data generator
train_data_gen = data_gen(data=traindata, batch_size=batch_size)

# Define the number of training steps
nb_train_steps = traindata.shape[0]//batch_size

# ...............>>>>>>>>>>>>>>>>
nb_train_steps = traindata.shape[0]
# ...............>>>>>>>>>>>>>>>>

print("Number of training and validation steps: {} and {}".format(nb_train_steps, len(valid_data)))

valid_data.shape[0]


# opt = RMSprop(lr=0.0001, decay=1e-6)
opt = Adam(lr=0.0001, decay=1e-5)
es = EarlyStopping(patience=5)
chkpt = ModelCheckpoint(filepath='best_model_todate', save_best_only=True, save_weights_only=True)
loss_object='binary_crossentropy'
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)
#batch_size = 16
batch_size = 1
#nb_epochs = 20
nb_epochs = 1
validation_steps=1
#callbacks=[es, chkpt]
callbacks = callbacks=[LambdaCallback(on_batch_end=lambda batch,logs:print(logs))]


history = model.fit( epochs=nb_epochs, steps_per_epoch=nb_train_steps,
                               validation_data=(x_test, y_test),
                               callbacks=[es, chkpt],
                               #validation_steps=1,
                               #class_weight={0:1.0, 1:0.4},
                               verbose=1)

#model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test), batch_size=64,callbacks=[es, chkpt],verbose=1 )


#-------------------------------------------------------

#Model 3
# Unbalanced datasets


model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", input_shape=(224,224,3)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(SeparableConv2D(128, (3,3), activation='relu', padding='same', name='Conv2_1'))
model.add(SeparableConv2D(128, (3,3), activation='relu', padding='same', name='Conv2_2'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_1'))
model.add(BatchNormalization(name='bn1'))
model.add(SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_2'))
model.add(BatchNormalization(name='bn2'))
model.add(SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_3'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_1'))
model.add(BatchNormalization(name='bn3'))
model.add(SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_2'))
model.add(BatchNormalization(name='bn4'))
model.add(SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_3'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(1024, activation='relu',name='dense1'))
model.add(Dropout(0.7,name='dropout6'))
model.add(Dense(512, activation='relu',name='dense2'))
model.add(Dropout(0.5,name='dropouut7'))
model.add(Dense(14, activation='softmax',name='dense3'))

model.summary()


print("[INFO] training w/ generator...")
H = model.fit_generator(
	trainGen,
	steps_per_epoch=NUM_TRAIN_IMAGES // BS,
	validation_data=testGen,
	validation_steps=NUM_TEST_IMAGES // BS,
	epochs=NUM_EPOCHS)



opt = Adam(lr=0.0001, decay=1e-5)
epc = 1
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


d=np.array[disease]

print(y_train[19])

model.fit(x_train, y_train, epochs=epc, validation_data=(x_test, y_test), batch_size=64)


#disease=["No Finding","Consolidation","Infiltration","Pneumothorax","Edema","Emphysema","Fibrosis","Effusion",
#         "Pneumonia","Pleural_Thickening","Cardiomegaly","Mass","Hernia","Atelectasis"]


#save model weights
#model.save_weights('D:/AKU Python Projects/Imran_Pneumonia/NIH_v1_model_weights.h5')

# Save model
#model.save('D:/AKU Python Projects/Imran_Pneumonia/NIH_v1__model.h5')

q='F:/Dataset2/NIHChest/images/valid/00030280_005.png'
img = image.load_img(q,target_size=(224,224,3))
img = image.img_to_array(img)
img = img/255
#classes = np.array(train.columns[2:])
proba = model.predict(img.reshape(1,224,224,3))
top_3 = np.argsort(proba[0])[:-4:-1]
print(top_3)
for i in range(3):
    print("{}".format(classes[top_3[i]])+" ({:.3})".format(proba[0][top_3[i]]))
plt.imshow(img)



#-------------------------------------------------------

#Model 4


model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", input_shape=(224,224,3)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(SeparableConv2D(128, (3,3), activation='relu', padding='same', name='Conv2_1'))
model.add(SeparableConv2D(128, (3,3), activation='relu', padding='same', name='Conv2_2'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_1'))
model.add(BatchNormalization(name='bn1'))
model.add(SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_2'))
model.add(BatchNormalization(name='bn2'))
model.add(SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_3'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_1'))
model.add(BatchNormalization(name='bn3'))
model.add(SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_2'))
model.add(BatchNormalization(name='bn4'))
model.add(SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_3'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(1024, activation='relu',name='dense1'))
model.add(Dropout(0.7,name='dropout6'))
model.add(Dense(512, activation='relu',name='dense2'))
model.add(Dropout(0.5,name='dropouut7'))
model.add(Dense(14, activation='softmax',name='dense3-model-4'))

model.summary()


print("[INFO] training w/ generator...")
H = model.fit_generator(
	trainGen,
	steps_per_epoch=NUM_TRAIN_IMAGES // BS,
	validation_data=testGen,
	validation_steps=NUM_TEST_IMAGES // BS,
	epochs=NUM_EPOCHS)



opt = Adam(lr=0.0001, decay=1e-5)
epc = 1
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

print(X_trainRosReshaped.shape)
print(Y_trainReshaped.shape)
print(X_testRosReshaped.shape)
print(Y_testReshaped.shape)


#d=np.array[disease]
#
#print(y_train[19])

model.fit(x_train, y_train, epochs=epc, validation_data=(x_test, y_test), batch_size=64)


#disease=["No Finding","Consolidation","Infiltration","Pneumothorax","Edema","Emphysema","Fibrosis","Effusion",
#         "Pneumonia","Pleural_Thickening","Cardiomegaly","Mass","Hernia","Atelectasis"]


#save model weights
#model.save_weights('D:/AKU Python Projects/Imran_Pneumonia/NIH_v1_model_weights.h5')

# Save model
#model.save('D:/AKU Python Projects/Imran_Pneumonia/NIH_v1__model.h5')

q='F:/Dataset2/NIHChest/images/valid/00030280_005.png'
img = image.load_img(q,target_size=(224,224,3))
img = image.img_to_array(img)
img = img/255
#classes = np.array(train.columns[2:])
proba = model.predict(img.reshape(1,224,224,3))
top_3 = np.argsort(proba[0])[:-4:-1]
print(top_3)
for i in range(3):
    print("{}".format(classes[top_3[i]])+" ({:.3})".format(proba[0][top_3[i]]))
plt.imshow(img)






# --------------------------------------------------------------
# Model 5 (From Chest_1_Working.py)
# VGG16

from keras.applications.vgg16 import VGG16
from keras.models import Model
weight_path = 'D:\Datasets\KaggleRandomCH\keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
im_size = 128
map_characters=dict_characters
#print(map_characters)

#def vgg16network(a,b,c,d,e,f,g):
a=X_trainRosReshaped[:10000]
b=Y_trainRosHot[:10000]
c=X_testRosReshaped[:3000]
d=Y_testRosHot[:3000]
e=class_weight
f=8
g=10

num_class = f
epochs = g
base_model = VGG16(#weights='imagenet',
    weights = weight_path, include_top=False, input_shape=(im_size, im_size, 3))
# Add a new top layer
x = base_model.output
x = Flatten()(x)
predictions = Dense(num_class, activation='softmax')(x)
# This is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
# First: train only the top layers (which were randomly initialized)
for layer in base_model.layers:
    layer.trainable = False
model.compile(loss='categorical_crossentropy', 
              optimizer=keras.optimizers.RMSprop(lr=0.0001), 
              metrics=['accuracy'])
callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]
model.summary()
model.fit(a,b, epochs=epochs, class_weight=e, validation_data=(c,d), verbose=1,callbacks = [MetricsCheckpoint('logs')])
score = model.evaluate(c,d, verbose=1)
print('\nKeras CNN #2 - accuracy:', score[1], '\n')
y_pred = model.predict(c)
print(num_class)
print(y_pred)
print('\n', sklearn.metrics.classification_report(np.where(d > 0)[1], np.argmax(y_pred, axis=1), target_names=list(map_characters.values())), sep='') 
Y_pred_classes = np.argmax(y_pred,axis = 1) 
print(y_pred)
print(Y_pred_classes)
print(len(Y_pred_classes))
Y_true = np.argmax(d,axis = 1) 
print(Y_true)
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
plotKerasLearningCurve()
plt.show()
plot_confusion_matrix(confusion_mtx, classes = list(map_characters.values()))
plt.show()




# --------------------------------------------------------------
# Model 6 (modified Model 1 copy to use balanced dataset ROS)

print(X_trainRosReshaped.shape)
print(Y_trainReshaped.shape)
print(X_testRosReshaped.shape)
print(Y_testReshaped.shape)


model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu',name='Conv2'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25,name='Dropout2'))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu",name='Conv3'))
model.add(MaxPooling2D(pool_size=(2, 2),name='max2'))
model.add(Dropout(0.25,name='Dropout4'))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu',name='Conv4'))
model.add(MaxPooling2D(pool_size=(2, 2),name='max3'))
model.add(Dropout(0.25,name='Dropout5'))
model.add(Flatten())
model.add(Dense(128, activation='relu',name='dense1'))
model.add(Dropout(0.5,name='dropout6'))
model.add(Dense(64, activation='relu',name='dense2'))
model.add(Dropout(0.5,name='dropouut7'))
#model.add(Dense(14, activation='sigmoid',name='dense3-Model-6'))
model.add(Dense(14, activation='softmax',name='dense3-Model-6'))

model.summary()

# With binary crossentropy - Use Sigmoid
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# With categorical crossentropy - Use Softmax
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


#d=np.array[disease]
#
#print(y_train[19])

#callbacks = [MetricsCheckpoint('logs')]

epc=2
history = model.fit(X_trainRosReshaped, Y_trainReshaped, epochs=epc, 
                    validation_data=(X_testRosReshaped, Y_testReshaped), 
                    callbacks = [MetricsCheckpoint('logs')], batch_size=64)


plot_learning_curve(history)
plt.show()
plotKerasLearningCurve()
plt.show()

# Mass
q='F:/Dataset2/NIHChest/images/valid/00030159_000.png'
#Pneumothorax
q='F:/Dataset2/NIHChest/images/valid/00030323_030.png'
#Atelectasis
q='F:/Dataset2/NIHChest/images/valid/00030300_002.png'

img = image.load_img(q,target_size=(224,224,3))
img = image.img_to_array(img)
img = img/255
#classes = np.array(train.columns[2:])
proba = model.predict(img.reshape(1,224,224,3))
top_3 = np.argsort(proba[0])[:-4:-1]
print(top_3)
for i in range(3):
    print("{}".format(classes[top_3[i]])+" ({:.3})".format(proba[0][top_3[i]]))
plt.imshow(img)

# ###############################################
#  Disease Key
    
#    NoFinding = "No Finding" #0
#    Consolidation="Consolidation" #1
#    Infiltration="Infiltration" #2
#    Pneumothorax="Pneumothorax" #3
#    Edema="Edema" # 4
#    Emphysema="Emphysema" #5
#    Fibrosis="Fibrosis" #6
#    Effusion="Effusion" #7
#    Pneumonia="Pneumonia" #8
#    Pleural_Thickening="Pleural_Thickening" #9
#    Cardiomegaly="Cardiomegaly" #10
#    Mass="Mass" #11
#    Hernia="Hernia" #12
#    Atelectasis="Atelectasis"  #13
    
# ###############################################




# ------------------------------------------------------------------
# Model 7 (From KaggleRandomCH_1.py)
# Using VGG16

from keras.applications.vgg16 import VGG16
from keras.models import Model
weight_path = 'D:\Datasets\KaggleRandomCH\keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
im_size = 128
map_characters=dict_characters
def vgg16network(a,b,c,d,e,f,g):
    num_class = f
    epochs = g
    base_model = VGG16(#weights='imagenet',
        weights = weight_path, include_top=False, input_shape=(im_size, im_size, 3))
    # Add a new top layer
    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(num_class, activation='softmax')(x)
    # This is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    # First: train only the top layers (which were randomly initialized)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(loss='categorical_crossentropy', 
                  optimizer=keras.optimizers.RMSprop(lr=0.0001), 
                  metrics=['accuracy'])
    callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]
    model.summary()
    model.fit(a,b, epochs=epochs, class_weight=e, validation_data=(c,d), verbose=1,callbacks = [MetricsCheckpoint('logs')])
    score = model.evaluate(c,d, verbose=0)
    print('\nKeras CNN #2 - accuracy:', score[1], '\n')
    y_pred = model.predict(c)
    print('\n', sklearn.metrics.classification_report(np.where(d > 0)[1], np.argmax(y_pred, axis=1), target_names=list(map_characters.values())), sep='') 
    Y_pred_classes = np.argmax(y_pred,axis = 1) 
    Y_true = np.argmax(d,axis = 1) 
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
    plotKerasLearningCurve()
    plt.show()
    plot_confusion_matrix(confusion_mtx, classes = list(map_characters.values()))
    plt.show()
    return model


vgg16network(X_trainRosReshaped[:10000], Y_trainRosHot[:10000], X_testRosReshaped[:3000], Y_testRosHot[:3000],class_weight,8,15)




# -------------------------------------------------------
# Model 8 (Copy of Model 2 with ROS data)


# Augmentation sequence 
seq = iaa.OneOf([
    iaa.Fliplr(), # horizontal flips
    iaa.Affine(rotate=20), # roatation
    iaa.Multiply((1.2, 1.5))]) #random brightness
    
    
    
#Training data generator
#Here I will define a very simple data generator. You can do more than this if you want but I think at this point,
# this is more than enough I need.
def data_gen(data, batch_size):
    #data=train_data
    #batch_size=1
    # Get total number of samples in the data
    n = len(data)
    steps = n//batch_size
    
    # Define two numpy arrays for containing batch data and labels
    batch_data = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)
    batch_labels = np.zeros((batch_size,2), dtype=np.float32)

    # Get a numpy array of all the indices of the input data
    indices = np.arange(n)
    
    # Initialize a counter
    i =0
    while True:
        np.random.shuffle(indices)
        # Get the next batch 
        count = 0
        next_batch = indices[(i*batch_size):(i+1)*batch_size]
        for j, idx in enumerate(next_batch):
            #print(j,idx)
            #j=1
            #idx=1
            img_name = data.iloc[idx]['image']
            label = data.iloc[idx]['label']
            
            # one hot encoding
            encoded_label = to_categorical(label, num_classes=2)
            # read the image and resize
            img = cv2.imread(str(img_name))
            img = cv2.resize(img, (224,224))
            
            # check if it's grayscale
            if img.shape[2]==1:
                img = np.dstack([img, img, img])
            
            # cv2 reads in BGR mode by default
            orig_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # normalize the image pixels
            orig_img = img.astype(np.float32)/255.
            
            batch_data[count] = orig_img
            #print(batch_data[1])
            batch_labels[count] = encoded_label
            
            # generating more samples of the undersampled class
            #if label==0 and count < batch_size-2:
            if label==1 and count < batch_size-2:
                #print(label,'--augmenting..', end='')
                aug_img1 = seq.augment_image(img)
                aug_img2 = seq.augment_image(img)
                aug_img1 = cv2.cvtColor(aug_img1, cv2.COLOR_BGR2RGB)
                aug_img2 = cv2.cvtColor(aug_img2, cv2.COLOR_BGR2RGB)
                aug_img1 = aug_img1.astype(np.float32)/255.
                aug_img2 = aug_img2.astype(np.float32)/255.

                batch_data[count+1] = aug_img1
                batch_labels[count+1] = encoded_label
                batch_data[count+2] = aug_img2
                batch_labels[count+2] = encoded_label
                count +=2
                #print('1',end='')
                
            else:
                count+=1
            
            if count==batch_size-1:
                break
            
        i+=1
        yield batch_data, batch_labels
            
        if i>=steps:
            i=0


def build_model():
    input_img = Input(shape=(224,224,3), name='ImageInput')
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv1_1')(input_img)
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv1_2')(x)
    x = MaxPooling2D((2,2), name='pool1')(x)
    
    x = SeparableConv2D(128, (3,3), activation='relu', padding='same', name='Conv2_1')(x)
    x = SeparableConv2D(128, (3,3), activation='relu', padding='same', name='Conv2_2')(x)
    x = MaxPooling2D((2,2), name='pool2')(x)
    
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_1')(x)
    x = BatchNormalization(name='bn1')(x)
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_3')(x)
    x = MaxPooling2D((2,2), name='pool3')(x)
    
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_1')(x)
    x = BatchNormalization(name='bn3')(x)
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_2')(x)
    x = BatchNormalization(name='bn4')(x)
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_3')(x)
    x = MaxPooling2D((2,2), name='pool4')(x)
    
    x = Flatten(name='flatten')(x)
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dropout(0.7, name='dropout1')(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dropout(0.5, name='dropout2')(x)
    x = Dense(14, activation='softmax', name='fc3')(x)
    
    model = Model(inputs=input_img, outputs=x)
    return model
model =  build_model()
model.summary()




#log_dir="C:/TensorBoardLog/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") 
log_dir="C:/TensorBoardLog/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)




#We will initialize the weights of first two convolutions with imagenet weights,
# Open the VGG16 weight file
f = h5py.File('D:/chest_xray/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', 'r')

# Select the layers for which you want to set weight.

w,b = f['block1_conv1']['block1_conv1_W_1:0'], f['block1_conv1']['block1_conv1_b_1:0']
model.layers[1].set_weights = [w,b]

w,b = f['block1_conv2']['block1_conv2_W_1:0'], f['block1_conv2']['block1_conv2_b_1:0']
model.layers[2].set_weights = [w,b]

w,b = f['block2_conv1']['block2_conv1_W_1:0'], f['block2_conv1']['block2_conv1_b_1:0']
model.layers[4].set_weights = [w,b]

w,b = f['block2_conv2']['block2_conv2_W_1:0'], f['block2_conv2']['block2_conv2_b_1:0']
model.layers[5].set_weights = [w,b]

f.close()
model.summary()    


print(X_trainRosReshaped.shape)
print(Y_trainReshaped.shape)
print(X_testRosReshaped.shape)
print(Y_testReshaped.shape)



# Get a train data generator
train_data_gen = data_gen(data=X_trainRosReshaped, batch_size=batch_size)

# Define the number of training steps
nb_train_steps = X_trainRosReshaped.shape[0]//batch_size

# ...............>>>>>>>>>>>>>>>>
nb_train_steps = X_trainRosReshaped.shape[0]
# ...............>>>>>>>>>>>>>>>>

print("Number of training and validation steps: {} and {}".format(nb_train_steps, len(X_testRosReshaped)))

X_testRosReshaped.shape[0]


# opt = RMSprop(lr=0.0001, decay=1e-6)
opt = Adam(lr=0.0001, decay=1e-5)
es = EarlyStopping(patience=5)
chkpt = ModelCheckpoint(filepath='best_model_todate', save_best_only=True, save_weights_only=True)
loss_object='binary_crossentropy'
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)
batch_size = 16
#batch_size = 1
#nb_epochs = 20
nb_epochs = 1
validation_steps=1
callb=[es, chkpt]
#callb = callbacks=[LambdaCallback(on_batch_end=lambda batch,logs:print(logs))]


history = model.fit_generator( train_data_gen,epochs=nb_epochs, steps_per_epoch=nb_train_steps,
                               validation_data=(X_testRosReshaped, Y_testReshaped),
                               callbacks=callb,
                               #validation_steps=8,
                               #class_weight={0:1.0, 1:0.4},
                               verbose=1)

#model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test), batch_size=64,callbacks=[es, chkpt],verbose=1 )


#-------------------------------------------------------

# --------------------------------------------------------------
# Model 9 (modified Model 6 copy to use balanced dataset ROS)
# model parameters changed in this 

print(X_trainRosReshaped.shape)
print(Y_trainReshaped.shape)
print(X_testRosReshaped.shape)
print(Y_testReshaped.shape)


#log_dir="C:/TensorBoardLog/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") 
log_dir="C:/TensorBoardLog/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)


model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(224,224,3),kernel_regularizer=l2(0.0005)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',name='Conv2',kernel_regularizer=l2(0.0005)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25,name='Dropout2'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu",name='Conv3',kernel_regularizer=l2(0.0005)))
model.add(MaxPooling2D(pool_size=(2, 2),name='max2'))
model.add(Dropout(0.25,name='Dropout4'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',name='Conv4',kernel_regularizer=l2(0.0005)))
model.add(MaxPooling2D(pool_size=(2, 2),name='max3'))
model.add(Dropout(0.25,name='Dropout5'))
model.add(Flatten())
model.add(Dense(128, activation='relu',name='dense1'))
model.add(Dropout(0.5,name='dropout6'))
model.add(Dense(64, activation='relu',name='dense2'))
model.add(Dropout(0.5,name='dropouut7'))
#model.add(Dense(14, activation='sigmoid',name='dense3-Model-9'))
model.add(Dense(14, activation='softmax',name='dense3-Model-8'))

model.summary()

# With binary crossentropy - Use Sigmoid
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

epc=1
#opt = SGD(lr=0.01, momentum=0.9)
opt='adam'

# With categorical crossentropy - Use Softmax
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


#d=np.array[disease]
#
#print(y_train[19])

#callbacks = [MetricsCheckpoint('logs')]

history = model.fit(X_trainRosReshaped, Y_trainReshaped, epochs=epc, 
                    validation_data=(X_testRosReshaped, Y_testReshaped), 
                    callbacks = [MetricsCheckpoint(log_dir)], batch_size=64,
                    verbose=1)


plot_learning_curve(history)
plt.show()
plotKerasLearningCurve()
plt.show()


# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['acc'], label='train')
pyplot.plot(history.history['val_acc'], label='test')
pyplot.legend()
pyplot.show()


# Mass
q='F:/Dataset2/NIHChest/images/valid/00030159_000.png'
#Pneumothorax
q='F:/Dataset2/NIHChest/images/valid/00030323_030.png'
#Atelectasis
q='F:/Dataset2/NIHChest/images/valid/00030300_002.png'

img = image.load_img(q,target_size=(224,224,3))
img = image.img_to_array(img)
img = img/255
#classes = np.array(train.columns[2:])
proba = model.predict(img.reshape(1,224,224,3))
top_3 = np.argsort(proba[0])[:-4:-1]
print(top_3)
for i in range(3):
    print("{}".format(classes[top_3[i]])+" ({:.3})".format(proba[0][top_3[i]]))
plt.imshow(img)


#Describe new numpy arrays
#dict_characters = {1: 'Consolidation', 2: 'Infiltration', 
#        3: 'Pneumothorax', 4:'Effusion', 5: 'Nodule Mass', 6: 'Atelectasis', 7: "Other Rare Classes"}
dict_characters = {1: "No Finding",2: "Consolidation",3: "Infiltration",4: "Pneumothorax",5: "Edema",6: "Emphysema",
                   7: "Fibrosis",8: "Effusion",9: "Pneumonia",10: "Pleural_Thickening",11: "Cardiomegaly",12: "Mass",
                   13: "Hernia",14: "Atelectasis"}

map_characters=dict_characters
c=X_testRosReshaped
d=Y_testReshaped
num_class = 15


y_pred = model.predict(c)
print(num_class)
print(y_pred)
print('\n', sklearn.metrics.classification_report(np.where(d > 0)[1], np.argmax(y_pred, axis=1), target_names=list(map_characters.values())), sep='') 

Y_pred_classes = np.argmax(y_pred,axis = 1) 
Y_true = np.argmax(d,axis = 1) 
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
plotKerasLearningCurve()
plt.show()
classlist=list(map_characters.values())
plot_confusion_matrix(confusion_mtx, classlist )
plt.show()




# ###############################################
#  Disease Key
    
#    NoFinding = "No Finding" #0
#    Consolidation="Consolidation" #1
#    Infiltration="Infiltration" #2
#    Pneumothorax="Pneumothorax" #3
#    Edema="Edema" # 4
#    Emphysema="Emphysema" #5
#    Fibrosis="Fibrosis" #6
#    Effusion="Effusion" #7
#    Pneumonia="Pneumonia" #8
#    Pleural_Thickening="Pleural_Thickening" #9
#    Cardiomegaly="Cardiomegaly" #10
#    Mass="Mass" #11
#    Hernia="Hernia" #12
#    Atelectasis="Atelectasis"  #13
    
# ###############################################













# Helper Functions  Learning Curves and Confusion Matrix

from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

class MetricsCheckpoint(Callback):
    """Callback that saves metrics after each epoch"""
    def __init__(self, savepath):
        super(MetricsCheckpoint, self).__init__()
        self.savepath = savepath
        self.history = {}
    def on_epoch_end(self, epoch, logs=None):
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        np.save(self.savepath, self.history)


def plot_learning_curve(history):
    plt.figure(figsize=(8,8))
    plt.subplot(1,2,1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./accuracy_curve.png')
    #plt.clf()
    # summarize history for loss
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./loss_curve.png')
    
    
def plotKerasLearningCurve():
    plt.figure(figsize=(10,5))
    metrics = np.load('logs.npy')[()]
    filt = ['acc'] # try to add 'loss' to see the loss learning curve
    for k in filter(lambda x : np.any([kk in x for kk in filt]), metrics.keys()):
        l = np.array(metrics[k])
        plt.plot(l, c= 'r' if 'val' not in k else 'b', label='val' if 'val' in k else 'train')
        x = np.argmin(l) if 'loss' in k else np.argmax(l)
        y = l[x]
        plt.scatter(x,y, lw=0, alpha=0.25, s=100, c='r' if 'val' not in k else 'b')
        plt.text(x, y, '{} = {:.4f}'.format(x,y), size='15', color= 'r' if 'val' not in k else 'b')   
    plt.legend(loc=4)
    plt.axis([0, None, None, None]);
    plt.grid()
    plt.xlabel('Number of epochs')


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize = (5,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_learning_curve(history):
    plt.figure(figsize=(8,8))
    plt.subplot(1,2,1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./accuracy_curve.png')
    #plt.clf()
    # summarize history for loss
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./loss_curve.png')
    
        
    
    