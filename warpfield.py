from frame import Frame
from quaternion import Quaternion, DualQuaternion

import numpy as np
from sklearn.neighbors import KDTree
from numba import jit, njit, prange

class WarpField:
    def __init__(self, n):

        # nodes 0-th axis index guide
        # 0:8   --- DualQuaternion
        # 8     --- weight
        # 9:12  --- position

        self.nodes = np.zeros((12,n))

        self.nodes[0,:] = 1     # initialize DualQuaternion to 1 0 0 0  0 0 0 0
        self.nodes[8,:] = 1     # initialize weight to 1
        self.nodes[9:12,:] = 0  # initialize node position to 0 0 0

        self.kdtree = KDTree(self.nodes[9:12, :].T, 500)
        self.k = 1

    def _kdtree(self, X):
        _, ind = self.kdtree.query(X.T, k=self.k)
        return ind.T

    def to_array(self, exclude_pos=True):
        n = 9 if exclude_pos else 12

        return self.nodes[:n].T.reshape(-1,1)

    def _fit(self, x, n, ind, frame):
        x = x.reshape(-1,1)
        nodes = self.nodes[:,ind]

        dist = np.sum(np.power(np.tile(x, (1,self.k)) - nodes[9:], 2), axis=0).reshape(1,-1)
        weight = nodes[8].reshape(1,-1)
        weight_ex = np.exp(-dist / (2*weight))
        
        nodes_dq = nodes[:8].reshape(8,-1)
        nodes_dqs = np.sum(nodes_dq * weight_ex, axis=1).reshape(-1,1)
        nodes_dqh_norm = np.sqrt(np.sum(nodes_dqs*nodes_dqs,axis=0))
        nodes_dqh = nodes_dqs / nodes_dqh_norm
        
        dq = DualQuaternion(*nodes_dqh)
        
        x_ = frame.world2cam(dq.transform(x))
        x2 = np.array([1,0,0]).reshape(-1,1) #frame.pix2cam(frame.cam2pix(x_))
        
        if np.isnan(x2[2]):
            return None
            
        n_ = dq.rotation.rotate(n)
        n_ = frame.world2cam(n, rotate_only=True)

        cost = np.sum(n_*(x_-x2))

        print(np.hstack([x_, x2]))

        dweight = dist/(2*weight*weight)

        ddq = np.array([np.hstack([np.identity(8), nodes_dq[:,i].reshape(-1,1)*dweight[:,i]]) * weight_ex[:,i] for i in range(self.k)])
        ddqh = np.identity(8)/nodes_dqh_norm - np.dot(nodes_dqh,nodes_dqh.T) / np.power(nodes_dqh_norm, 3)
       
        dT1 = dq.d_transform(x)
        dT2 = np.hstack([dq.rotation.d_rotate(n), np.zeros((3,4))])
        
        j1 = np.dot(dT1, ddqh)
        j2 = np.dot(dT2, ddqh)

        jacobian = np.array([np.sum(np.dot(j1, ddq[i])*n_ + np.dot(j2,ddq[i])*(x_-x2), axis=0) for i in range(self.k)])


        print(jacobian)
        
        return cost, jacobian


    def optimize(self, X, N, frame):
        # compute kdtree and normals

        samples = X.shape[1]
        cost = np.zeros((samples, 1))
        jacobian = np.zeros((samples, self.nodes.shape[1]*9))

        ind = self._kdtree(X)

        for k in range(50):

            for i in range(samples):
                cost[i], jac_frag = self._fit(X[:,i], N[:,i], ind[:,i], frame)
                for j in range(self.k):
                    jacobian[i, ind[j,i]*9:ind[j,i]*9+9] = jac_frag[j]

            #print(jacobian)

            old_array = self.to_array()
            new_array = old_array + np.dot(np.linalg.pinv(jacobian), cost)

            print(new_array)

            self.nodes[:9] = new_array.reshape(-1, 9).T

            #print(np.mean(new_array - old_array))


if __name__ == "__main__":
    #print(WarpField(100)._fit(np.array([0,0,1]), np.array([1,0,0]), [3,2,1], Frame(0)))

    frame0 = Frame(0)
    frame1 = Frame(100)
    
    X = frame0.cam2world(frame0.pixel_3d_cam())
    N = frame0.cam2world(frame0.pixel_3d_cam_normal(), rotate_only = True)

    ind = (np.sum(X,axis=0) == 0)
    X = X[:,~ind][:,400:500]
    N = N[:,~ind][:,400:500]

    X = np.array([0,0,0]).reshape(-1,1)
    N = np.array([1,0,0]).reshape(-1,1)

    WarpField(1).optimize(X,N, frame1)
