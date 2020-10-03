import numpy as np
import cv2
from numba import jit, njit, prange
from math import sqrt
from frame import Frame
#from warpfield import WarpField
#from pykinect_load import *

def homogenous(x):
    return np.vstack([x, np.ones(x.shape[1])])

@jit(nopython=True)
def bilinear(x, voxel, doprint=False):
    a, b, c = map(int, [x[0], x[1], x[2]])
    a_, b_, c_ = min(a+1, voxel.shape[0]), min(b+1, voxel.shape[1]), min(c+1, voxel.shape[2])

    t,u,v = a_-x[0], b_-x[1], c_-x[2]
    t_,u_,v_ = 1-t, 1-u, 1-v


    if doprint:
        print(t, u, v, voxel[a][b][c], voxel[a][b][c_], voxel[a][b_][c], voxel[a_][b_][c_], voxel[a_][b][c], voxel[a_][b][c_], voxel[a][b_][c], voxel[a_][b_][c_])

    return (t*u*v*voxel[a][b][c] + t*u*v_*voxel[a][b][c_] + t*u_*v*voxel[a][b_][c] + t*u_*v_*voxel[a][b_][c_]
    + t_*u*v*voxel[a_][b][c] + t_*u*v_*voxel[a_][b][c_] + t_*u_*v*voxel[a_][b_][c] + t_*u_*v_*voxel[a_][b_][c_])


@jit(nopython=True)
def nearest(x, voxel, doprint=False):
    return voxel[round(x[0])][round(x[1])][round(x[2])]


@jit(nopython=True)
def project(pose, x):
    return np.array([pose[0,0]*x[0]+pose[0,1]*x[1]+pose[0,2]*x[2]+pose[0,3], 
            pose[1,0]*x[0]+pose[1,1]*x[1]+pose[1,2]*x[2]+pose[1,3],
            pose[2,0]*x[0]+pose[2,1]*x[1]+pose[2,2]*x[2]+pose[2,3]])

@jit(nopython=True)
def matmul(pose, x):
    return np.array([pose[0,0]*x[0]+pose[0,1]*x[1]+pose[0,2]*x[2], 
            pose[1,0]*x[0]+pose[1,1]*x[1]+pose[1,2]*x[2],
            pose[2,0]*x[0]+pose[2,1]*x[1]+pose[2,2]*x[2]])

@jit(nopython=True)
def norm(x):
    return sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2])

@jit(nopython=True)
def get_normal(x, voxel):
    h = 0.5
    xp = [min(voxel.shape[0], x[0]+h), min(voxel.shape[1], x[1]+h), min(voxel.shape[2], x[2]+h)]
    xm = [max(0, x[0]-h), max(1, x[1]-h), max(2, x[2]-h)]

    gradient = np.array([-bilinear([xp[0], x[1], x[2]], voxel) + bilinear([xm[0], x[1], x[2]], voxel),
                        -bilinear([x[0], xp[1], x[2]], voxel) + bilinear([x[0], xm[1], x[2]], voxel),
                        -bilinear([x[0], x[1], xp[2]], voxel) + bilinear([x[0], x[1], xm[2]], voxel)])

    return gradient / norm(gradient)

@njit(parallel = True)
def raycast(h, w, pose, K_inv, voxel, voxel_size, voxel_min, a):
    cam_pos = np.array([pose[0,3], pose[1,3], pose[2,3]])
    epsilon = 0.01
    iter_n = 40

    img = np.ones((h, w, 3))*256

    for i in prange(h*w):
        px_point = [i%w, int(i/w), 1]
        ray = project(pose, matmul(K_inv, px_point)) - cam_pos
        dist = 1.
        travel = 0.5

        for j in range(iter_n):
            x = ray * travel * a + cam_pos
            x = (x - voxel_min)/voxel_size

            dist = min(bilinear(x, voxel), dist)
            travel = travel + dist

            if dist < epsilon:
                # get normal
                normal = get_normal(x, voxel)
                
                #img[px_point[1], px_point[0]] = (normal * 128 + 128)# * (iter_n - j) / iter_n

                img[px_point[1], px_point[0]] = normal[1] * 128 + 128 # * (iter_n - j) / iter_n

                break


    return img

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

    voxel_min = np.zeros((3,1))
    voxel_max = np.zeros((3,1))

    for frame_number in range(1,2):
        frame = Frame(frame_number)
        w,h = frame.w, frame.h


        """
        V = frame.pixel_3d_cam()
        writePLY('rgbd.ply', np.hstack([V.T, img_rgb.reshape(-1,3)]))
        """

        # area for voxel volume
        # get voxel frustrum
        frustum = np.matmul(frame.K_inv, np.array([[0, 0, 0], [0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]]).astype(float).T)
        frustum = np.concatenate([[0], [np.max(frame.img_d)] * 4]) * frustum
        frustum = np.matmul(frame.pose, homogenous(frustum))[:3]

        voxel_min = np.min(np.hstack([voxel_min, frustum]), axis=1)
        voxel_max = np.max(np.hstack([voxel_max, frustum]), axis=1)

    # create voxel space
    voxel_size = 0.005
    voxel_shape = ((voxel_max - voxel_min) / voxel_size).astype(int)[:3]

    voxel = np.ones(voxel_shape)
    voxel_col = np.zeros((*voxel_shape, 3))
    voxel_weight = np.zeros(voxel_shape)

    voxel_coords_ind = np.concatenate([x.reshape(1, -1) for x in np.meshgrid(*[range(i) for i in voxel_shape],indexing='ij')], axis=0)
    voxel_coords = voxel_coords_ind * voxel_size + voxel_min.reshape(3,1)

    print(voxel_coords_ind.shape)



    for frame_number in range(2):
        frame = Frame(frame_number)
        

        #   TODO : DYNAMIC FUSION
        #   after computing WarpField, transform Frame0 cam coordinates into Frame_n cam coordinates using WarpField
        #   Frame0 cam2world -> WarpField ------------------------------------------------------------------------> Derivative -> Optimization
        #                                 -> Frame_n world2cam-> Frame_n cam2pix -> round -> pix2cam2world /     -> (after convergence) Get PSDF   -> Fuse


        # convert to cam, pixel coord
        voxel_coords_cam = frame.world2cam(voxel_coords)
        voxel_coords_pix = frame.cam2pix(voxel_coords_cam)

        pix_x = voxel_coords_pix[0]
        pix_y = voxel_coords_pix[1]


        # only consider voxels inside the pixel range
        valid_ind = np.logical_and(pix_x == np.clip(pix_x, 0, w), pix_y == np.clip(pix_y, 0, h))
        
        voxel_z = voxel_coords_cam[2, valid_ind]
        voxel_depth = frame.img_d[pix_y[valid_ind].astype(int), pix_x[valid_ind].astype(int)]

        # find voxels within depth +- error range
        a = voxel_size * 5

        depth_delta = voxel_depth - voxel_z
        valid_ind2 = np.logical_and(depth_delta < a, depth_delta > -a)
        ind = voxel_coords_ind.T[valid_ind][valid_ind2]

        col = frame.img_rgb[pix_y[valid_ind][valid_ind2].astype(int), pix_x[valid_ind][valid_ind2].astype(int)]

        # update voxel values
        voxel_picks = voxel[ind[:, 0], ind[:, 1], ind[:, 2]]
        voxel_col_picks = voxel_col[ind[:,0], ind[:,1], ind[:,2]]
        voxel_weight_picks = voxel_weight[ind[:, 0], ind[:, 1], ind[:, 2]]

        
        voxel[ind[:, 0], ind[:, 1], ind[:, 2]] = (voxel_weight_picks * voxel_picks + depth_delta[valid_ind2] / a) / (voxel_weight_picks+1)
        voxel_col[ind[:,0], ind[:,1], ind[:,2]] = (voxel_weight_picks.reshape(-1,1) * voxel_col_picks + col) / (voxel_weight_picks+1).reshape(-1,1)
        voxel_weight[ind[:, 0], ind[:, 1], ind[:, 2]] = voxel_weight_picks + 1

        print(frame_number)


    valid_ind = (voxel_weight > 0).reshape(-1)
    print(valid_ind.shape, ind.shape)
    ind = voxel_coords_ind.T[valid_ind]

    v = (voxel[ind[:,0], ind[:,1], ind[:,2]] + 1) / 2
    v3 = np.tile(v.reshape(-1, 1), (1, 3))

    writePLY("tsdf.ply", np.concatenate([ind, v3 * 255], axis=1))
    writePLY("rgb_tsdf.ply", np.concatenate([ind , voxel_col[ind[:,0], ind[:,1], ind[:,2]]], axis=1))
    
    raycasted_img = raycast(frame.h,frame.w, frame.pose, frame.K_inv, voxel,voxel_size, voxel_min, a)

    cv2.imshow('image', cv2.cvtColor(raycasted_img.astype(np.uint8), cv2.COLOR_RGB2BGR))
    #cv2.imshow('image', cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

