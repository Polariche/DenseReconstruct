import numpy as np
import cv2

def homogenous(x):
    return np.vstack([x, np.ones(x.shape[1])])

class Frame:

    def __init__(self, frame_number):
        self.K = np.loadtxt('data/camera-intrinsics.txt')
        self.K_inv = np.linalg.inv(K)
        self.img_rgb = cv2.cvtColor(cv2.imread('data/frame-%06d.color.jpg' % frame_number), cv2.COLOR_BGR2RGB)
        self.img_d = cv2.imread('data/frame-%06d.depth.png' % frame_number, -1).astype(float)
        self.img_d = img_d / 1000.

        self.pose = np.loadtxt('data/frame-%06d.pose.txt' % frame_number)
        self.pose_inv = np.linalg.inv(pose)

        self.h, self.w = img_d.shape

        

    def world2cam(self, x):
        return np.dot(self.pose_inv, homogeneous(x))

    def cam2world(self, x):
        return np.dot(self.pose, homogeneous(x))

    def cam2pix(self, x):
        y = np.dot(self.K, x)
        return (y/y[2])[:2]

    def pix2cam(self, x):
        x = np.round(x)
        u, v = x[1], x[0]

        y = np.dot(self.K_inv, x)
        y[2] = img_d.reshape(-1)[u*self.w + v]
        y[:2] *= y[2]

        return y

    def pixel_3d_cam(self, x):
        n = self.w*self.h
        
        u = np.array([[i%self.w, int(i/self.w)] for i in range()]).T # raw pixel index; shape is (2, h*w) 
        u_ = homogenous(u)   # homogeneous

        # pixels in camera coordinate; shape is (3, h*w)
        V = img_d.reshape(-1) * np.matmul(K_inv, u_)

        return V

