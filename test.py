import numpy as np
def bilinear(x, voxel, doprint=False):
    a, b, c = map(int, [x[0], x[1], x[2]])
    a_, b_, c_ = min(a+1, voxel.shape[0]), min(b+1, voxel.shape[1]), min(c+1, voxel.shape[2])

    t,u,v = a_-x[0], b_-x[1], c_-x[2]
    t_,u_,v_ = 1-t, 1-u, 1-v


    if doprint:
        print(t, u, v, voxel[a][b][c], voxel[a][b][c_], voxel[a][b_][c], voxel[a_][b_][c_], voxel[a_][b][c], voxel[a_][b][c_], voxel[a][b_][c], voxel[a_][b_][c_])

    return (t*u*v*voxel[a][b][c] + t*u*v_*voxel[a][b][c_] + t*u_*v*voxel[a][b_][c] + t*u_*v_*voxel[a][b_][c_]
    + t_*u*v*voxel[a_][b][c] + t_*u*v_*voxel[a_][b][c_] + t_*u_*v*voxel[a_][b_][c] + t_*u_*v_*voxel[a_][b_][c_])
print(bilinear(np.array([0.0976002380000125, 0.9930539526893227, 0.2184972158974361]), np.ones((2,2,2)), True))