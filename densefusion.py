import numpy as np
import cv2
from numba import njit, prange
from pykinect_load import *

def homogenous(x):
    return np.vstack([x, np.ones(x.shape[1])])

def writePLY(filename, X):

    X[:, 3:] = X[:, 3:].astype(np.uint8)

    ply_file = open(filename,'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n"%(X.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("end_header\n")

    for i in range(X.shape[0]):
        ply_file.write("%f %f %f %d %d %d\n" % (X[i,0], X[i,1], X[i,2], X[i,3],X[i,4],X[i,5]))


if __name__ == "__main__":
    # intrinsics
    K = np.loadtxt('data/camera-intrinsics.txt')
    K_inv = np.linalg.inv(K)

    # read images
    frame_number = 1

    img_rgb = cv2.cvtColor(cv2.imread('data/frame-%06d.color.jpg' % frame_number), cv2.COLOR_BGR2RGB)
    img_d = cv2.imread('data/frame-%06d.depth.png' % frame_number, -1).astype(float)
    img_d = img_d / 1000.


    pose = np.loadtxt('data/frame-%06d.pose.txt' % frame_number)

    # pixel coordinates
    h,w = img_d.shape[:2]
    n = h*w

    u = np.array([[i%w, int(i/w)] for i in range(n)]).T # raw pixel index; shape is (2, h*w) 
    u_ = homogenous(u)   # homogeneous

    # pixels in camera coordinate; shape is (3, h*w)
    V = img_d.reshape(-1) * np.matmul(K_inv, u_)

    writePLY('rgbd.ply', np.hstack([V.T, img_rgb.reshape(-1,3)]))


    # area for voxel volume
    # get voxel frustrum
    frustum = np.matmul(K_inv, np.array([[0, 0, 0], [0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]]).astype(float).T)
    frustum = np.concatenate([[0], [np.max(img_d)] * 4]) * frustum
    frustum = np.matmul(pose, homogenous(frustum))[:3]

    voxel_min = np.min(frustum, axis=1)
    voxel_max = np.max(frustum, axis=1)

    voxel_size = 0.02
    voxel_shape = ((voxel_max - voxel_min) / voxel_size).astype(int)[:3]

    voxel = np.ones(voxel_shape)
    voxel_coords_ind = np.concatenate([x.reshape(1, -1) for x in np.meshgrid(*[range(i) for i in voxel_shape],indexing='ij')], axis=0).T
    voxel_coords = voxel_coords_ind * voxel_size + voxel_min

    voxel_coords_cam = np.matmul(np.linalg.inv(pose), homogenous(voxel_coords.T))[:3]

    voxel_coords_pix = np.matmul(K, voxel_coords_cam)
    voxel_coords_pix = (voxel_coords_pix / voxel_coords_pix[2])[:2]

    pix_x = voxel_coords_pix[0]
    pix_y = voxel_coords_pix[1]

    # only consider voxels inside the pixel range
    valid_ind = np.logical_and(pix_x == np.clip(pix_x, 0, w), pix_y == np.clip(pix_y, 0, h))
    
    voxel_z = voxel_coords_cam[2, valid_ind]
    voxel_depth = img_d[pix_y[valid_ind].astype(int), pix_x[valid_ind].astype(int)]

    # find voxels within depth +- error range
    a = 0.1
    depth_delta = voxel_depth - voxel_z
    valid_ind2 = np.logical_and(depth_delta < a, depth_delta > -a)
    ind = voxel_coords_ind[valid_ind][valid_ind2]
    voxel[ind[:, 0], ind[:, 1], ind[:, 2]] = depth_delta[valid_ind2] / a

    v = (voxel[ind[:, 0], ind[:, 1], ind[:, 2]] + 1) / 2
    v3 = np.tile(v.reshape(-1, 1), (1, 3))

    #print(np.concatenate([ind, v3], axis=1))
    writePLY("tsdf.ply", np.concatenate([ind, v3 * 255], axis=1))

    writePLY("rgb_tsdf.ply", np.concatenate([ind , img_rgb[pix_y[valid_ind][valid_ind2].astype(int), pix_x[valid_ind][valid_ind2].astype(int)] * v3], axis=1))




