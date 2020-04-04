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


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
 
        return self.score
 
def _sigmoid(x):
    return 1. / (1. + np.exp(-x))
 
def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5
    boxes = []
    netout[..., :2]  = _sigmoid(netout[..., :2])
    netout[..., 4:]  = _sigmoid(netout[..., 4:])
    netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh
 
    for i in range(grid_h*grid_w):
        row = i / grid_w
        col = i % grid_w
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[int(row)][int(col)][b][4]
            if(objectness.all() <= obj_thresh): continue
            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[int(row)][int(col)][b][:4]
            x = (col + x) / grid_w # center position, unit: image width
            y = (row + y) / grid_h # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height
            # last elements are class probabilities
            classes = netout[int(row)][col][b][5:]
            box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
            boxes.append(box)
    return boxes
 
def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    new_w, new_h = net_w, net_h
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3

def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h
    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    union = w1*h1 + w2*h2 - intersect
    return float(intersect) / union
 
def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        print("Length of boxes: ", len(boxes))
        nb_class = len(boxes[0].classes)
        print("Number of classes i: ", nb_class)
        print("Example box classes:", boxes[0].classes)
    else:
        return
    for c in range(nb_class): #nb_classes
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        print("Length of sorted indices: ", len(sorted_indices))
        print("Sorted Indices: ", sorted_indices)
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0: continue
            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0
# get all of the results above a threshold
def get_boxes(boxes, labels, thresh):
    v_boxes = list()#, v_labels, v_scores = list(), list(), list()
    # enumerate all boxes
    for box in boxes:
        # enumerate all possible labels
        for i in range(len(labels)):
            # check if the threshold for this label is high enough
            if box.classes[i] > thresh:
                v_boxes.append(box)
                #v_labels.append(labels[i])
                #v_scores.append(box.classes[i]*100)
                # don't break, many labels may trigger for one box
    return v_boxes#, v_labels, v_scores

def plot_scatter(boxes):
    x = [ (i.xmin + i.xmax)/2 for i in boxes]
    y = [ (i.ymin + i.ymax)/2 for i in boxes]
    pyplot.scatter(x,y)
    pyplot.show()

class UserModel:
    def __init__(self, sinPath, cosPath, probPath):
        self.session = tf.Session()
        keras.backend.set_session(self.session)
        if not sinPath == "":
            self.sinModel = load_model(sinPath)
            self.sinModel._make_predict_function()
        if not cosPath == "":
            self.cosModel = load_model(cosPath)
            self.cosModel._make_predict_function()
        if not probPath == "":
            self.probModel = load_model(probPath)
            self.probModel._make_predict_function()
        self.anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
        self.WIDTH, self.HEIGHT = 416, 416
        self.class_threshold = 0.7
        self.image_size = 0
        self.labels = ["person", "bicycle", "car", "motorbike", "bus", "truck", "tree"]

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
        image_w, image_h, oogabooga = img.shape
        image = np.asarray(Image.fromarray(img).resize((self.WIDTH,self.HEIGHT))) # CHECK THIS
        image = image.astype('float32')
        image /= 255.0
        # add a dimension so that we have one sample
        start1 = time.time()
        start = time.time()
        image = expand_dims(image, 0)
        with self.session.as_default():
            with self.session.graph.as_default():
                yhat = self.probModel.predict(image)

        end = time.time()
        print("PREDICTING TOOK {} SECONDS".format(end - start))
        start = time.time()
        boxes = list()
        for i in range(len(yhat)):
            # decode the output of the network
            boxes += decode_netout(yhat[i][0], self.anchors[i], self.class_threshold, self.HEIGHT, self.WIDTH)

        # correct the sizes of the bounding boxes for the shape of the image
        print("BOXES:", len(boxes))
        end = time.time()
        print("DECODING TOOK {} SECONDS".format(end-start))
        start = time.time()
        correct_yolo_boxes(boxes, image_h, image_w, self.HEIGHT, self.WIDTH)
        end = time.time()
        print("CORRECTING YOLO BOXES TOOK {} SECONDS".format(end-start))

        # suppress non-maximal boxes
        start = time.time()
        do_nms(boxes, 0.5)
        end = time.time()
        print("SUPPRESSING NON MAXIMAL BOXES TOOK {} SECONDS".format(end-start))

        # get the details of the detected objects
        start = time.time()
        v_boxes = get_boxes(boxes, self.labels, self.class_threshold)
        end = time.time()
        print("GETTING THE BOXES TOOK {} SECONDS".format(end-start))#, v_labels, v_scores = get_boxes(boxes, self.labels, self.class_threshold)
        end1 = time.time()
        print("ALL THE BOXING COMPUTATION TOOK {} SECONDS".format(end1-start1))
        return v_boxes, image_w, image_h #, v_labels, v_scores, image_h, image_w

    def predictProb(self, img, count):
        v_boxes, image_w, image_h = self.predict_image(img)
        start = time.time()
        imp = self.importance_img((image_w, image_h), v_boxes, 10, 1.5, math.pi/2, 3, 1.1)
        end = time.time()
        print("ALL OF THE IMAGE IMPORTANCE MAPPING TOOK {} SECONDS".format(end - start))
        return imp


