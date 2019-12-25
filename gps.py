# from pyparrot.Bebop import Bebop
# from pyparrot.DroneVisionGUI import DroneVisionGUI
# import threading
# import cv2
# import time
# import face_recognition
# import numpy as np
# from PIL import Image
# import queue
# from scipy.spatial import distance
# import math
#
# import argparse
# import logging
# coding:UTF-8
from math import radians, cos, sin, asin, sqrt, atan2, degrees

class gps:
    def __init__(self, drone):
        self.loop = True
        self.drone = drone
        # self.a1 = 500
        # self.a2 = 500
        self.direct = 0
        self.pitch_rate = 0
        self.yaw_rate = 0
        self.vertical_rate = 0
        self.move = 0

    def initial_bearing(self, pointA, pointB):

        if (type(pointA) != tuple) or (type(pointB) != tuple):
            raise TypeError("Only tuples are supported as arguments")

        lat1 = radians(pointA[0])
        lat2 = radians(pointB[0])

        diffLong = radians(pointB[1] - pointA[1])

        x = sin(diffLong) * cos(lat2)
        y = cos(lat1) * sin(lat2) - (sin(lat1)
                                     * cos(lat2) * cos(diffLong))

        initial_bearing = atan2(x, y)

        initial_bearing = degrees(initial_bearing)
        compass_bearing = (initial_bearing + 360) % 360

        return compass_bearing

    def get_rate(self, _point, isVertical):

        isChanged = False   # self.change가 변경되었는지 체크
        loop = True

        if self.move == 0:
            self.a1 = self.drone.sensors.sensors_dict['GpsLocationChanged_latitude']
            self.a2 = self.drone.sensors.sensors_dict['GpsLocationChanged_longitude']
            self.direct = self.initial_bearing((self.a1, self.a2), _point)

            # if self.a1 == 500 or self.a2 == 500:
            #     self.drone.safe_land(5)

            # if self.direct > 180:
            #     self.direct = self.direct - 360

            print(self.direct)
            if self.a2 - _point[1] > 0.00005:
                print("ggoggogogoogo")
                self.drone.safe_takeoff(5)
                self.drone.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=60, duration=3.5)
                self.move = 1

        else:
            self.a1 = self.drone.sensors.sensors_dict['GpsLocationChanged_latitude']
            self.a2 = self.drone.sensors.sensors_dict['GpsLocationChanged_longitude']
            self.direct = self.initial_bearing((self.a1, self.a2), _point)
            print('a1 : ', self.a1)
            print('a2 : ', self.a2)
            # if self.a1 == 500 or self.a2 == 500:
            #     self.drone.safe_land(5)

            # if self.direct > 180:
            #     self.direct = self.direct - 360


            if isVertical == False:
                if abs(self.a2 - _point[1]) > 0.000065: # 35M 범위에 못 미침
                    self.yaw_rate = 0
                    self.pitch_rate = 20
                    self.vertical_rate = 0
                    print("vertical false straight")
                    print('남은 거리 : ', abs(self.a2 - _point[1]))
                elif abs(self.a2 - _point[1]) <= 0.000065: # 35M 범위안에 들음
                    self.drone.fly_direct(roll=0, pitch=0, yaw=45, vertical_movement=0, duration=14.5)
                    self.pitch_rate = 0
                    self.yaw_rate = 0
                    self.vertical_rate = 0
                    isChanged = True
                    loop = False
                    print("vertical false turn right")
                    print('남은 거리 : ', abs(self.a2 - _point[1]))

            elif isVertical == True:
                if abs(self.a1 - _point[0]) > 0.00012: # 35M 범위에 못 미침
                    self.yaw_rate = 0
                    self.pitch_rate = 20
                    self.vertical_rate = 0
                    print("straight")
                    print('남은 거리 : ', abs(self.a1 - _point[0]))
                elif abs(self.a1 - _point[0]) <= 0.000012: # 35M 범위안에 들음
                    self.drone.fly_direct(roll=0, pitch=0, yaw=45, vertical_movement=0, duration=16)
                    self.pitch_rate = 0
                    self.yaw_rate = 0
                    self.vertical_rate = 0
                    isChanged = True
                    print("turn right")
                    print('남은 거리 : ', abs(self.a1 - _point[0]))


            else: # 필요 없을듯? 가면 안되는 상태
                self.pitch_rate = 0
                self.yaw_rate = 0
                self.vertical_rate = 0

        return self.yaw_rate, self.vertical_rate, self.pitch_rate, isChanged



