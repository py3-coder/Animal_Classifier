#!/usr/bin/env python
# coding: utf-8

# # Animal_Classifier

# ## Cat Vs Dog Image Classification 

# ### @Author :Saurabh ... Date :10-Jan

# ### Importing all required lib

# In[1]:


import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings("ignore")
from pathlib import Path
from imutils import paths
import cv2


# In[2]:


import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.models import Sequential , Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D,MaxPooling2D,Dropout,Input ,Lambda
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from glob import glob
from PIL import Image
from IPython.display import Image


# - ### Checking wheather physical_devices for tranings images

# In[4]:


tf.config.list_physical_devices()


# In[5]:


pwd


# In[6]:


path = 'E:\\DataScience\\Data_Center\\Animal_Classifier'
os.listdir(path)


# In[7]:


dataset =path+"\\dataset"
os.listdir(dataset)


# In[8]:


train_path =dataset+"\\training_set"
test_path =dataset+"\\test_set"


# In[9]:


os.listdir(train_path)


# In[10]:


args ={}
args['dataset'] =dataset
args['train_path'] =train_path
args['test_path'] =test_path


# In[11]:


args


# In[12]:


def count_images(list):
    lst = list
    count_cat = 0
    count_dogs = 0
    for img in lst:
        if img == 'cats':
            count_cat = count_cat+1
        else :
            count_dogs = count_dogs+1
    print("Count Cat_images:",count_cat)
    print("Count Dogs_images:",count_dogs)


# In[13]:


def data_lables(list):
    ipaths =list
    data  =[]
    labels=[]
    for iPath in ipaths:
        label=iPath.split(os.path.sep)[-2]
        image=cv2.imread(iPath)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image=cv2.resize(image,(224,224))
        data.append(image)
        labels.append(label)
    
    ##data=np.array(data)/255.0
    labels=np.array(labels)
    return labels


# In[14]:


print("Tranings images Counts :")
lst1=list(paths.list_images(args['train_path']))
lst2=list(data_lables(lst1))
count_images(lst2)


# In[15]:


print("Testing/Validation images Counts :")
lst1=list(paths.list_images(args['test_path']))
lst2=list(data_lables(lst1))
count_images(lst2)


# In[16]:


print("Total images of Cat = Train images + Validation images :",4000+1000)
print("Total images of Dogs = Train images + Validation images :",4000+1000)


# In[17]:


train_path


# In[18]:


os.listdir(train_path)


# ## Cats

# In[24]:


cats =[]
images_path =train_path+"\\cats"
for filenames in os.listdir(images_path):
    cats.append(os.path.join(images_path,filenames))
    
fig = plt.figure(figsize=(20,20))
for i in range(10):
    idx =np.random.randint(0,100)
    plt.subplot(5,5,i+1)
    image =plt.imread(cats[idx])
    plt.imshow(image)


# ## Dogs

# In[25]:


dogs =[]
images_path =train_path+"\\dogs"
for filenames in os.listdir(images_path):
    dogs.append(os.path.join(images_path,filenames))
    
fig = plt.figure(figsize=(20,20))
for i in range(10):
    idx =np.random.randint(0,100)
    plt.subplot(5,5,i+1)
    image =plt.imread(dogs[idx])
    plt.imshow(image)


# ## Cat Vs Dog

# In[27]:


fig = plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
image1 =plt.imread(cats[3])
plt.imshow(image1)
plt.title("Cat")
plt.subplot(1,2,2)
image2 =plt.imread(dogs[14])
plt.imshow(image2)
plt.title("Dog")


# In[28]:


fig = plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
image1 =plt.imread(cats[24])
plt.imshow(image1)
plt.title("Cat")
plt.subplot(1,2,2)
image2 =plt.imread(dogs[12])
plt.imshow(image2)
plt.title("Dog")


# In[23]:


fig = plt.figure(figsize=(12,12))
plt.subplot(1,2,1)
image1 =plt.imread(cats[13])
plt.imshow(image1)
plt.title("Cat")
plt.subplot(1,2,2)
image2 =plt.imread(dogs[77])
plt.imshow(image2)
plt.title("Dog")


# In[29]:


for dirname,_ ,filesnames in os.walk(dataset):
    for filename in filesnames:
        print(os.path.join(dirname,filename))


# ## Trainings images using VGG19

# In[30]:


Image_Size=[224,224]
batch_size=32


# In[31]:


#importing the VGG19 and also removing top_layers and we will use pre-trained weight(imagnet weight)
mobilenet =VGG19(include_top=False,input_shape=Image_Size +[3],weights='imagenet')


# In[32]:


#need_not to train the weight a we are using pre trained weight
for layer in mobilenet.layers:
    layer.trainable = False


# In[33]:


##
folder= glob(train_path+"\\*")
folder


# In[34]:


#adding output layer
X =Flatten()(mobilenet.output)
out_layers =Dense(len(folder),activation='softmax')(X)


# In[35]:


#Model summary
model =Model(inputs=mobilenet.input,outputs=out_layers)
model.summary()


# In[36]:


#model_compile
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[37]:


#Image agumentation
#image agumentation for training set
train_gen = ImageDataGenerator(rescale=1./255,
                               rotation_range=10,
                               width_shift_range=.1,
                               height_shift_range=.1,
                               shear_range=0.1,
                               zoom_range=0.1,
                               horizontal_flip=True,
                               fill_mode="nearest")

#image agumentation for validation set
val_gen =ImageDataGenerator(rescale=1./255,
                            rotation_range=10,
                            width_shift_range=.1,
                            height_shift_range=.1,
                            shear_range=0.1,
                            zoom_range=0.1,
                            horizontal_flip=True,
                            fill_mode="nearest")


# In[38]:


##
train_data =train_gen.flow_from_directory(train_path,target_size=(224,224),
                                             batch_size=32,
                                             classes=['cats','dogs'],
                                             class_mode='categorical')


valid_data =val_gen.flow_from_directory(test_path,target_size=(224,224),
                                        batch_size=32,
                                        classes=['cats','dogs'],
                                        class_mode='categorical')


# In[39]:


#traning images 
with tf.device('/GPU:0'):
    history =model.fit_generator(train_data,
                                 validation_data=valid_data,
                                 epochs=30,
                                 steps_per_epoch=len(train_data),
                                 validation_steps=len(valid_data)
)


# In[40]:


#save the model.... 
model.save("Model.h5")


# In[41]:


pwd


# In[42]:


path='E:\\DataScience\\Data_Center\\Animal_Classifier'
os.listdir(path)


# In[43]:


os.listdir(dataset)


# ### Test_Image for model testing

# In[44]:


Test_images_path=dataset+"\\Test_data"


# In[45]:


os.listdir(Test_images_path)


# In[57]:


lst =[]
for dirname,_ ,filesnames in os.walk(Test_images_path):
    for filename in filesnames:
        lst.append(os.path.join(dirname,filename))


# In[58]:


lst


# In[60]:


fig = plt.figure(figsize=(20,20))
for i in range(5):
    idx =np.random.randint(0,5)
    plt.subplot(5,5,i+1)
    image =plt.imread(lst[idx])
    plt.imshow(image)


# ##### Test:01 
# - Cat

# In[61]:


from tensorflow.keras.preprocessing import image 
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
img =image.load_img('E:\\DataScience\\Data_Center\\Animal_Classifier\\dataset\\Test_data\\cat_test01.jpg',target_size=(224,224))
#X=image.img_to_array(img)
img=np.expand_dims(img,axis=0)
#img_data=preprocess_input(X)
result=model.predict(img)

if result[0][0] == 1:
    print("Cat")
elif result[0][1] == 1:
    print("Dog")


# In[62]:


Image='E:\\DataScience\\Data_Center\\Animal_Classifier\\dataset\\Test_data\\cat_test01.jpg'
image=plt.imread(Image)
plt.imshow(image)
plt.title("Actual : Cat \n" +"Predicted  : Cat")


# In[93]:


from tensorflow.keras.preprocessing import image 
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img

def test_image(path):
    img =image.load_img(path,target_size=(224,224))
    #X=image.img_to_array(img)
    img=np.expand_dims(img,axis=0)
    #img_data=preprocess_input(X)
    result=model.predict(img)
    if result[0][0] == 1:
        return("Cat")
    elif result[0][1] == 1:
        return("Dog")


# In[96]:


from tensorflow.keras.preprocessing import image 
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
path='E:\\DataScience\\Data_Center\\Animal_Classifier\\dataset\\Test_data\\cat_test02.jpg'
output = test_image(path)
image=plt.imread(path)
plt.imshow(image)
plt.title("Actual : Cat \n" +"Predicted  :"+output)


# In[94]:


from tensorflow.keras.preprocessing import image 
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
path='E:\\DataScience\\Data_Center\\Animal_Classifier\\dataset\\Test_data\\dog_test03.jpg'
output = test_image(path)
image=plt.imread(path)
plt.imshow(image)
plt.title("Actual : Dog \n" +"Predicted  : "+ output )


# In[97]:


from tensorflow.keras.preprocessing import image 
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
path='E:\\DataScience\\Data_Center\\Animal_Classifier\\dataset\\Test_data\\dog_test02.jpg'
output = test_image(path)
image=plt.imread(path)
plt.imshow(image)
plt.title("Actual : Dog \n" +"Predicted  : "+ output )


# In[98]:


from tensorflow.keras.preprocessing import image 
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
path='E:\\DataScience\\Data_Center\\Animal_Classifier\\dataset\\Test_data\\dog_test01.jpg'
output = test_image(path)
image=plt.imread(path)
plt.imshow(image)
plt.title("Actual : Dog \n" +"Predicted  : "+ output )


#  # !!! All Test image are Correctly Predicted .....
#  

# In[ ]:




