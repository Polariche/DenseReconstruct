from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

import numpy as np

class Kinect:
    def __init__(self):
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth)

    def readColor(self):
        if self._kinect.has_new_color_frame():
            return self._kinect.get_last_color_frame()
        return None

    def readDepth(self):
        if self._kinect.has_new_depth_frame():
            return self._kinect.get_last_depth_frame()
        return None

if __name__ == "__main__":
    kinect = Kinect()

    #print(kinect.has_new_color_frame())
    print(kinect.readColor())