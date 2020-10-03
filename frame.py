import numpy as np
import cv2

def homogenous(x):
    return np.vstack([x, np.ones(x.shape[1])])

class Frame:

    def __init__(self, frame_number):
        self.K = np.loadtxt('VolumeDeformData/minion/data/colorIntrinsics.txt')[:3,:3]
        self.K_inv = np.linalg.inv(self.K)
        self.img_rgb = cv2.cvtColor(cv2.imread('VolumeDeformData/minion/data/frame-%06d.color.png' % frame_number), cv2.COLOR_BGR2RGB)
        self.img_d = cv2.imread('VolumeDeformData/minion/data/frame-%06d.depth.png' % frame_number, -1).astype(float)
        self.img_d = self.img_d / 1000.

        self.pose = np.identity(4) #np.loadtxt('data/frame-%06d.pose.txt' % frame_number)
        self.pose_inv = np.linalg.inv(self.pose)

        self.h, self.w = self.img_d.shape

        

    def world2cam(self, x, rotate_only=False):
        x = x.reshape(3,-1)

        if rotate_only:
            return np.dot(self.pose_inv[:3,:3], x)
        else:
            return np.dot(self.pose_inv, homogenous(x))[:3]

    def cam2world(self, x, rotate_only=False):
        x = x.reshape(3,-1)

        if rotate_only:
            return np.dot(self.pose[:3,:3], x)
        else:
            return np.dot(self.pose, homogenous(x))[:3]

    def cam2pix(self, x):
        x = x.reshape(3,-1)

        with np.errstate(divide='ignore', invalid='ignore'):
            y = np.dot(self.K, x)
            return (y/y[2])[:2]

    def pix2cam(self, x):
        x = x.reshape(2,-1)

        x = np.round(x).astype(np.uint8)
        u, v = x[1], x[0]

        y = np.dot(self.K_inv, homogenous(x))
        y[2] = self.img_d.reshape(-1)[u*self.w + v]
        y[:2] *= y[2]

        return y

    def normal(self, x, start_from_cam=False):
        # assume world coordinate
        if not start_from_cam:
            x = self.world2cam(x)
        x = self.cam2pix(x)
        # fetch depth
        y = self.pix2cam(x)

        du = np.array([[1], [0]])
        dv = np.array([[0], [1]])

        y_u1 = self.pix2cam(x+du)
        y_v1 = self.pix2cam(x+dv)

        v1 = y_u1 - y
        v2 = y_v1 - y

        n = np.vstack([v1[1]*v2[2]-v1[2]*v2[1], v1[2]*v2[0]-v1[0]*v2[2], v1[0]*v2[1]-v1[1]*v2[0]])
        n = n / np.sqrt(np.sum(n*n, axis=0)).reshape(1,-1)

        return n

    def pixel_3d_cam(self):
        n = self.w*self.h
        
        u = np.array([[i%self.w, int(i/self.w)] for i in range(n)]).T # raw pixel index; shape is (2, h*w) 
        u_ = homogenous(u)   # homogeneous

        # pixels in camera coordinate; shape is (3, h*w)
        V = self.img_d.reshape(-1) * np.matmul(self.K_inv, u_)

        return V

    def pixel_3d_cam_normal(self):
        V = self.pixel_3d_cam()
        V_u1 = np.hstack([V[:,1:],V[:,0].reshape(3,1)]) 
        V_v1 = np.hstack([V[:,self.w:],V[:,:self.w]])

        v1 = V_u1 - V
        v2 = V_v1 - V

        # n = normalize(cross(v1, v2))

        n = np.vstack([v1[1]*v2[2]-v1[2]*v2[1], v1[2]*v2[0]-v1[0]*v2[2], v1[0]*v2[1]-v1[1]*v2[0]])
        n = n / np.sqrt(np.sum(n*n, axis=0)).reshape(1,-1)

        return n



