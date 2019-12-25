from pyparrot.Bebop import Bebop
from pyparrot.DroneVisionGUI import DroneVisionGUI
import tensorflow as tf
from pyparrot.DroneVision import DroneVision

import threading
import cv2
import time
# import face_recognition
import numpy as np
# from PIL import Image
import queue
from scipy.spatial import distance
import math
#
import argparse
import logging
# from tf_pose.estimator import TfPoseEstimator
# from tf_pose.networks import get_graph_path, model_wh

import gps

# import detect
# import yolov3_model
import drone_model_sumup as c3dModel
from keras import backend as K
'''
global variables
'''
fps_time = 0
q = queue.Queue(10)
globals_imgBuffer = []
globals_res = []

#   trained model for 3d CNN
c3dmodel = c3dModel.modelCompile()


class UserVision:
    def __init__(self, vision, drone):
        # self.yv3 = yolov3_model.Yolov3()
        self.gps = gps.gps(drone)
        self.loop = True
        self.index = 0
        self.vision = vision
        self.drone = drone
        self.x =0
        self.y =0
        self.a =0
        self.df = 0
        self.size = 0
        self.pitch_rate = 0
        self.yaw_rate = 0
        self.vertical_rate = 0

        self.dst =0
        self.drone_centroid = (int(856 / 2), int(480 * (0.4)))
        self.change = -1

        self.point_index = 0

        # -1 : windowdetect.get_rate
        # -2 : gps.get_rate
        # 0 : posedetect.get_rate
        # 1 : facedetect.get_rate


    def set_p_y_v(self, p, y, v):
        self.pitch = p
        self.yaw = y
        self.vertical = v

    def get_p_y_v(self):
        return self.pitch, self.yaw, self.vertical

    def get_loop(self):
        return self.loop

    def detect_target(self, args):
        frame = self.vision.get_latest_valid_picture()
        result = None
        if frame is not None:
            result = self.yv3.run_model(frame)
        if result is not None:
            pitch, yaw, vertical = self.yv3.get_pitch_yaw_vertical()
            self.set_p_y_v(pitch, yaw, vertical)

            info = "test"
            cv2.putText(result, text=info, org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255, 0, 0), thickness=2)
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("result", result)
            print("THREAD!!! Pitch:{}, Yaw:{}, Vertical:{}".format(pitch, yaw, vertical))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.loop = False

            q.put((pitch, yaw, vertical, self.loop))
            time.sleep(0.0005)
        else:
            pass

    def get_frames(self, args):
        img = self.vision.get_latest_valid_picture()
        if (img is not None):
            # img write test
            # cv2.imwrite(str(self.index)+'.jpg', img)
            # print(str(self.index))
            global globals_imgBuffer, globals_res
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert color: bgr->rgb
            # img = img[(480//9):(480//9)*8, (856//16):(856//16)*15]  # 가로=856, 세로=480 -> 16:9 비율에 맞게 크롭
            img = cv2.resize(img, (171, 128))
            if len(globals_imgBuffer) < 16:
                globals_imgBuffer.append(img)
            else:  # full!!
                globals_res = np.array(globals_imgBuffer)
                globals_res = np.moveaxis(globals_res, 0, 2)
                globals_res = globals_res[np.newaxis]

                # print('np.shape(globals_res)', np.shape(globals_res))
                result = c3dModel.violencePred(model=c3dmodel, input=globals_res)
                print(result)

                # cv2.putText(self.vision.get_latest_valid_picture(), text=result, org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                #             fontScale=1, color=(255, 0, 0), thickness=2)
                # cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
                # cv2.imshow("result", self.vision.get_latest_valid_picture())
                #
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     self.loop = False

                globals_imgBuffer = []
        # self.index += 1
'''
end of UserVision class
'''




def flying_drone(bebopVision, args):


    print("Press 'q' if you want to stop and land drone")

    loop = True
    drone = args[0]
    status = args[1]
    q = args[2]

    if status == 't':
        testFlying = True
    else :
        testFlying = False

    if (testFlying):
        drone.safe_takeoff(5)
        print("take_off done!")

        isVertical = False

        isChanged = False

        point_index = 0

        loop = True

        _POINT_LIST = []
        # _POINT1 = (36.0100677056262, 129.3213531517531)
        # _POINT1 = (36.009831, 129.321545)
        _POINT1 = (36.009998, 129.321326)
        _POINT2 = (36.010475, 129.322042)
        _POINT3 = (36.010069, 129.322466)
        _POINT4 = (36.009626, 129.321714)
        _POINT_LIST.append(_POINT1)
        _POINT_LIST.append(_POINT2)
        _POINT_LIST.append(_POINT3)
        _POINT_LIST.append(_POINT4)


        gpsk = gps.gps(drone)

        yaw_rate, vertical_rate, pitch_rate, isChanged = gpsk.get_rate(_POINT_LIST[point_index], isVertical)
        q.put((pitch_rate, yaw_rate, vertical_rate, loop))

        params = q.get()
        drone.fly_direct(roll=0, pitch=params[0], yaw=params[1], vertical_movement=params[2], duration=0.5)
        print("Main Fn: {}\t{}\t{}".format(params[0], params[1], params[2]))

        print("==========isChange============( ", isChanged, " )============isChange==========")

        # 가로 세로 구분 없이 한것
        while True:
            if isChanged == True:

                if isVertical == True:
                    isVertical = False
                else:
                    isVertical = True

                point_index += 1
                if point_index >= 4:
                    bebop.safe_land(10)
                    print('다 돌았다.')
                    break

            yaw_rate, vertical_rate, pitch_rate, isChanged = gpsk.get_rate(_POINT_LIST[point_index], isVertical)
            q.put((pitch_rate, yaw_rate, vertical_rate, loop))

            params = q.get()
            drone.fly_direct(roll=0, pitch=params[0], yaw=params[1], vertical_movement=params[2], duration=0.5)
            print("Main Fn: {}\t{}\t{}".format(params[0], params[1], params[2]))

    print("disconnecting")
    drone.disconnect()




if __name__ == "__main__":
    # config = tf.ConfigProto()
    # config.gpu_optionst.allow_growth = True
    # make my bebop object
    bebop = Bebop()


    # connect to the bebop
    success = bebop.connect(5)
    bebop.safe_land(3)
    status = input("Input 't' if you want to TAKE OFF or not : ")
    # yolnir = Yolnir(droneVision)
    droneVision = DroneVisionGUI(bebop, is_bebop=True, user_code_to_run=flying_drone,
                                 user_args=(bebop, status, q))

    userVision = UserVision(droneVision, bebop)
    droneVision.set_user_callback_function(userVision.get_frames, user_callback_args=None)
    droneVision.open_video()


