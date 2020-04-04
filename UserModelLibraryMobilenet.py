import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from keras.models import load_model
from numpy import expand_dims
from keras.preprocessing.image import load_img, img_to_array
import cv2
import math
from scipy.integrate import quad
from PIL import Image
import time
import random
import argparse


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, score = -1, label = -1):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.label = label
        self.score = score

    def get_label(self):
        return self.label

    def get_score(self):
        return self.score

def plot_scatter(boxes):
    x = [ (i.xmin + i.xmax)/2 for i in boxes]
    y = [ (i.ymin + i.ymax)/2 for i in boxes]
    pyplot.scatter(x,y)
    pyplot.show()

class UserModel:
    def __init__(self, sinPath, cosPath, probPath, protoPath):
        self.session = tf.Session()
        keras.backend.set_session(self.session)
        if not sinPath == "":
            self.sinModel = load_model(sinPath)
            self.sinModel._make_predict_function()
        if not cosPath == "":
            self.cosModel = load_model(cosPath)
            self.cosModel._make_predict_function()
        self.probModel = probPath
        self.protoModel = protoPath
        self.WIDTH, self.HEIGHT = 300, 300
        self.class_threshold = 0.7
        self.predCLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
        self.CLASSES = ["bicycle", "bird", "boat","bus", "car", "cat", "dog", "motorbike", "person", "pottedplant", "train"]

    def function_x(self, x, width,maxi, imp, theta):
        w = width/2
        b = math.sqrt(4*math.log(imp))/w
        coef = w/2.5
        temp = maxi*(math.e**(-1*(b*(x-w - coef*math.cos(theta)))**2))
        return temp

    def function_y(self, y, h, maxi, i_e, i_m):
        C = maxi/(i_e * i_m)
        a = h/(4*C*(i_e*i_m - 1))
        B = (1 + math.sqrt(1 - (i_e - 1)/(i_e*i_m - 1)))/(2*a)
        A = -1*(B**2/(4*C*(i_e*i_m - 1)))
        return A*y**2 + B*y + C

    def importance_box(self, size, bBox, maxi, imp_x, theta, imp_y_e, imp_y_m):
        x1 = quad(lambda x: self.function_x(x, size[0], maxi, imp_x, theta ), bBox.xmin, bBox.xmax)
        x2 = quad(lambda x: self.function_x(x, size[0], maxi, imp_x, theta ), 0, size[0])
        x = x1[0]/x2[0]
        y1 = quad(lambda y: self.function_y(y, size[1], maxi, imp_y_e, imp_y_m ), bBox.ymin, bBox.ymax)
        y2 = quad(lambda y: self.function_y(y, size[1], maxi, imp_y_e, imp_y_m ), 0, size[1])
        y = y1[0]/y2[0]
        return min(x, y)
    def importance_img(self, size, v_boxes, maxi, imp_x, theta, imp_y_e, imp_y_m):
        c = 0
        for box in v_boxes:
            c += self.importance_box(size, box, maxi, imp_x, theta, imp_y_e, imp_y_m)**1.5
        return c**0.666
      
    def predictVelocity(self, img, count): #theta not implemented yet
        img = np.asarray(Image.fromarray(img).resize((128,128)))
        img = np.asarray([img])
        start = time.time()
        with self.session.as_default():
            with self.session.graph.as_default():
                cos = self.cosModel.predict(img).tolist()[0][0]
                sin = self.sinModel.predict(img).tolist()[0][0]
                angle = math.degrees(math.atan(sin/cos))
        end = time.time()
        print("VELOCITY PREDICTION TOOK {} SECONDS".format(end - start))
        return angle

    def predict_image(self, img):
        h, w, oogabooga = img.shape
        image = np.asarray(Image.fromarray(img).resize((self.WIDTH,self.HEIGHT))) # CHECK THIS
        #image = image.astype('float32')
        #image /= 255.0
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and
        # predictions
        print("[INFO] loading model...")
        net = cv2.dnn.readNetFromCaffe(self.protoModel, self.probModel)
        print("[INFO] computing object detections...")
        net.setInput(blob)
        detections = net.forward()
        # add a dimension so that we have one sample
        v_boxes = []
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > self.class_threshold:
                # extract the index of the class label from the `detections`,
                # then compute the (x, y)-coordinates of the bounding box for
                # the object
                idx = int(detections[0, 0, i, 1])
                if ( self.predCLASSES[idx] not in self.CLASSES):
                    continue
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                print((startX, startY, endX, endY))
                try:
                    bBox = BoundBox(startX, startY, endX, endY, confidence, self.predCLASSES[idx])
                except:
                    bBox = BoundBox(startX, startY, endX, endY, confidence, None)
                v_boxes.append(bBox)
                # display the prediction
                try:
                    label = "{}: {:.2f}%".format(self.predCLASSES[idx], confidence * 100)
                except:
                    label = 'DIDNT WORK'
                print("[INFO] {}".format(label))
        
        return v_boxes, w, h #, v_labels, v_scores, image_h, image_w

    def predictProb(self, img, count):
        v_boxes, image_w, image_h = self.predict_image(img)
        start = time.time()
        imp = self.importance_img((image_w, image_h), v_boxes, 10, 1.5, math.pi/2, 3, 1.1)
        end = time.time()
        print("ALL OF THE IMAGE IMPORTANCE MAPPING TOOK {} SECONDS".format(end - start))
        return imp, v_boxes


