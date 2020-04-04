import cv2
import os

path = "/home/anant/Pictures/Drone"
img = cv2.imread(path + "/" + os.listdir(path)[0])
image = cv2.resize(img, (128,128))
cv2.imshow("window", image)
cv2.waitKey(12000)
