#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 09:41:53 2022

@author: sayantanmanna
"""

#from configparser import Interpolation
#from fileinput import filename
#from tkinter.tix import IMAGE
#from turtle import width
import numpy as np
import pickle
import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras import models, layers, preprocessing
#import cv2
#import os
#import io




#loaded_model = pickle.load(open('/Users/sayantanmanna/Desktop/cc_class/desa_boiwl.sav', 'rb'))


###def loaded_image(image_file):
	#img = Image.open(image_file)
	#return img

#def scale(image):
   #image = tf.cast(image, tf.float32)
   #image /=255.0

   #return tf.image.resize(image, [256,256])

#def decode_img(image):
   #img = tf.image.decode_jpeg(image, channels=3)
   #img = scale(img)
   #return np.expand_dims(img, axis=0)

#def saveImage(byteImage):
    #bytesImg = io.BytesIO(byteImage)
    #imgFile = Image.open(bytesImg)   
   
    #return imgFile


class_names = ['algae',
                     'balanus',
                     'blue_mussel',
                     'christmas_tree_worm',
                     'finger_sponge',
                     'gooseneck_barnacle',
                     'kelp',
                     'rock_oysters',
                     'stinging_hydrozoan',
                     'zebra_mussel']


def load_model():
   model = tf.keras.models.load_model('keras_model.hdf5')
   return model

def predict(img, model):
   size = (256,256)
   image = ImageOps.fit(img, size, Image.ANTIALIAS)
   img_array = tf.keras.preprocessing.image.img_to_array(image)
   img_array = tf.expand_dims(img_array, 0)
    
   predictions = model.predict(img_array)
    
   predicted_class = class_names[np.argmax(predictions[0])]
   confidence = round(100 * (np.max(predictions[0])), 2)
   return predicted_class, confidence


def import_predict(image_data, model):

   size = (256,256)
   image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
   img = np.asarray(image)
   image_reshape = img[np.newaxis,...]
   prediction = model.predict(image_reshape)
   return prediction


def new_predict(image, model):

   data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
   # Replace this with the path to your image
   #resize the image to a 224x224 with the same strategy as in TM2:
   #resizing the image to be at least 224x224 and then cropping from the center
   size = (224, 224)
   image = ImageOps.fit(image, size, Image.ANTIALIAS)

   #turn the image into a numpy array
   image_array = np.asarray(image)
   # Normalize the image
   normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
   # Load the image into the array
   data[0] = normalized_image_array

   # run the inference
   prediction = model.predict(data)
   predicted_class = class_names[np.argmax(prediction[0])]
   confidence = round(100 * (np.max(prediction[0])), 2)
   return predicted_class, confidence

def main():
    
   
   st.title('Coral Image Classification')

   image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

   if image_file is not None:
	
       image = Image.open(image_file)
       st.image(image, use_column_width=True)
       model = load_model()
       predictions = new_predict(image, model)
	
       if(predictions[1]>65):
       	 st.success('Predicted class is : {}'.format(predictions[0]))
         st.success('confidence is : {}'.format(predictions[1]))
	 #st.success('Confidence percentage is : {}'.format(predictions[1]))
       else:
       	 st.success('Sorry!!!!Please upload only coral picture, unable to identify any other image')
		
	 
if __name__ == '__main__':
    main()
    
    
    
