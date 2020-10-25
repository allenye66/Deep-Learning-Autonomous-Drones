# Created by Allen Ye and Anant Bhatia from Lynbrook High School

# Autonomous-Drone-Delivery-Program-COVID-19
 
View our paper and project summary here: https://drive.google.com/drive/u/3/folders/1IlKEk60zxq8dQnekGxOkYXngBuomAFQA


Here is a conglomeration of file depcting the code we wrote to create an autonomous drone using a CNN-LSTM model to aid in food and package delivery during the 2020 quarantine.

Steering Angle Dataset Exploration: Here is where we explored methods in making our CNN-LSTM predictor, as well as coded the final version. We also have graphs for the results of our code. We also define the Gaussian and Edge detection preprocessing functions over here.

Yolov3 Bounding Boxes: Here is where we created a transfer learning model from the Yolov3 architecture to find bounding boxes of cars, people, and trees in our images. These bounding boxes were used by our probability model to calculate the probability of collision.

Weight determination functions: Here is where we defined the functions user to calculate the probability of colliding into any given object. The final probability determination function can be found in the Yolov3 script, as well as the UserModelLibrary scripts.

Data Exploration: Here is where we explored the data intially given to us, and found that the data was abnormally distributed. This helped us deermine the wraparound problem, as well as why our models prediction were near 0 in the early stages of the process.

Trial.py: This is the script to fly the actualy drone.

UserModelLibraries: This is the final conglomeration of all of our code - the probability functions, models, and pre/post processing function used to run our algorithms.

All pictures and graphs are also included in the pictures and graphs photo.
