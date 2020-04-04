from pyparrot.Bebop import Bebop
from pyparrot.DroneVision import DroneVision
import threading
import cv2
import time
from enum import Enum, auto
from UserModelLibraryMobilenet import UserModel, BoundBox
import time

class Model(Enum):
    BEBOP = auto()
    MAMBO = auto()
    ANAFI = auto()

class UserVision:
    def __init__(self, vision):
        self.index = 0
        self.vision = vision
        self.font = cv2.FONT_HERSHEY_SIMPLEX 
        self.fontScale = 1
        self.color = (255, 255, 255) 
        self.thickness = 2
        self.userModel = UserModel("/home/anant/pyparrot/Drone Models/model_sin_1.h5", "/home/anant/pyparrot/Drone Models/model_cos_1.h5", '/home/anant/pyparrot/Drone Models/object-detection-deep-learning/MobileNetSSD_deploy.caffemodel', '/home/anant/pyparrot/Drone Models/object-detection-deep-learning/MobileNetSSD_deploy.prototxt.txt')
        self.count = 0
        self.fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.vw = 0

    def streamVideo(self, args):
        #print("saving picture")
        start = time.time()
        self.count += 1
        img = self.vision.get_latest_valid_picture()
        if(self.vw == 0):
            self.vw = cv2.VideoWriter('trial.avi', self.fourcc, 15, 1232640)

        imp , v_boxes = self.userModel.predictProb(img, self.count)
        self.preprocessing(img, self.userModel.predictVelocity(img, self.count), imp, v_boxes)#, self.userModel.predictVelocity(img, self.count), self.userModel.predictProb(img, self.count))#self.userModel.predictProb(img))

        if(img is not None):
            
            cv2.imshow("window", img)#cv2.imwrite(filename, img)
            cv2.waitKey(1)
            vw.write(img)
            self.index +=1
            end = time.time()
            print("SHOWING THE IMAGE TOOK {} SECONDS".format(end - start))
        else:
            print("Img is none bitch")
            #filename = "/home/anant/Pictures/test_image_%06d.png" % self.index
        
  
    def preprocessing(self, img, velocity, prob, v_boxes):
        org = (50, 50)
        cv2.putText(img, 'Angle is: {}'.format(velocity), org, self.font, self.fontScale, self.color, self.thickness, cv2.LINE_AA)
        org = (50, 150)
        cv2.putText(img, 'Probability is: {}%'.format(prob), org, self.font, self.fontScale, self.color, self.thickness, cv2.LINE_AA)
        for box in v_boxes:
            cv2.rectangle(img, (box.xmin, box.ymin), (box.xmax, box.ymax), (0,0,0), 2)


# make my bebop object
bebop = Bebop()

# connect to the bebop
success = bebop.connect(5)

if (success):
    bebopVision = DroneVision(bebop, Model.BEBOP, buffer_size = 1)
    userVision = UserVision(bebopVision)
    bebopVision.set_user_callback_function(userVision.streamVideo, user_callback_args=None)
    success = bebopVision.open_video()
    if (success):
        print("Vision successfully started!")
        #removed the user call to this function (it now happens in open_video())
        #bebopVision.start_video_buffering()

        # skipping actually flying for safety purposes indoors - if you want
        # different pictures, move the bebop around by hand
        print("Fly me around by hand!")
        bebop.smart_sleep(5)

        bebop.smart_sleep(120)
        print("Finishing demo and stopping vision")
        bebopVision.close_video()
        uservision.vw.release()

  # disconnect nicely so we don't need a reboot
    bebop.disconnect()
else:
    print("Error connecting to bebop.  Retry")