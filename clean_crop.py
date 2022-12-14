from tensorflow.keras.applications.xception import Xception,preprocess_input,decode_predictions
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4' 
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img,img_to_array,array_to_img
import numpy as np
import math
import matplotlib.cm as cm
import cv2
from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

class crop:
    def __init__(self):
        with suppress_stdout():
            self.model = Xception(weights='imagenet')
        self.model.layers[-1].activation = None
        self.last_layer = "block14_sepconv2_act"
        self.dims = (299,299)
        self.preprocess = preprocess_input
        self.decoder = decode_predictions

    def to_array(self, path, dims):
        im = load_img(path,target_size=dims)
        im = img_to_array(im)
        im = np.expand_dims(im,axis=0)
        return im

    def create_heatmap(self,im):
        grad_model = Model([self.model.inputs],[self.model.get_layer(self.last_layer).output,self.model.output])

        with tf.GradientTape() as tape:
            output,pred = grad_model(im)
            index = tf.argmax(pred[0])
            channel = pred[:,index]
        grads = tape.gradient(channel, output)
        pooled = tf.reduce_mean(grads,axis=(0,1,2))
        output = output[0]
        heatmap = output @ pooled[...,tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap,0)/tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    def convole(self,im):
        colors,count = np.unique(im.reshape(-1,im.shape[-1]), axis=0, return_counts=True)
        return colors[count.argmax()]
    def crop(self,im,path):
        imgray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        imgray = cv2.blur(imgray, (15, 15))
        ret, thresh = cv2.threshold(imgray, math.floor(np.average(imgray)), 255, cv2.THRESH_BINARY_INV)
        dilated = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))
        dilated = dilated.astype(np.uint8)
        (contours, _) = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        new_contours = []
        for c in contours:
            if cv2.contourArea(c) < 4000000:
                new_contours.append(c)
        best_box = [-1, -1, -1, -1]
        for c in new_contours:
            x, y, w, h = cv2.boundingRect(c)
            if best_box[0] < 0:
                best_box = [x, y, x + w, y + h]
            else:
                if x < best_box[0]:
                    best_box[0] = x
                if y < best_box[1]:
                    best_box[1] = y
                if x + w > best_box[2]:
                    best_box[2] = x + w
                if y + h > best_box[3]:
                    best_box[3] = y + h
            cropped = im[best_box[1] : best_box[-1], best_box[0] : best_box[2]]
            
            cropped = array_to_img(cropped)
            cropped.save(f'{path}_clean.jpg')

    def get_heatmap(self,path,thres=50):
        res = img_to_array(load_img('images/stylised.png'))
        x,y,_=res.shape

        base = img_to_array(load_img(path,target_size=(x,y)))
    
        im = self.preprocess(self.to_array(path,self.dims))
        pool = self.convole(res)
        pred = self.model.predict(im,verbose=0)
        heatmap = self.create_heatmap(im)
        heatmap = np.uint8(255*heatmap)
        jet = cm.get_cmap('jet')
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        jet_heatmap = img_to_array(array_to_img(jet_heatmap).resize((y,x)))
        
        for i in range(x):
            for j in range(y):
                if jet_heatmap[i][j][0] < thres:
                    res[i][j] = pool
        self.crop(res,path)
        
        

        
