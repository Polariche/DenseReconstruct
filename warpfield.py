from frame import Frame
from quaternion import Quaternion, DualQuaternion

import numpy as np
from sklearn.neighbors import KDTree

class WarpField:
    def __init__(self, n):

        # nodes 0-th axis index guide
        # 0:8   --- DualQuaternion
        # 8     --- weight
        # 9:12  --- position

        self.nodes = np.zeros((12,n))

        self.nodes[0,:] = 1     # initialize DualQuaternion to 1 0 0 0  0 0 0 0
        self.nodes[8,:] = 1     # initialize position to 1


        self.kdtree = KDTree(self.nodes[9:12, :].T, 500)
        self.k = 3

    def kdtree(self, X):
        return self.kdtree.query(X.T, k=self.k).T

    def _fit(self, x, n, ind, frame):
        x = x.reshape(-1,1)
        nodes = self.nodes[:,ind]

        dist = np.sum(np.power(np.tile(x, (1,self.k)) - nodes[9:], 2), axis=0).reshape(1,-1)
        weight = nodes[8].reshape(1,-1)
        weight_ex = np.exp(-dist / (2*weight))
        
        nodes_dq = nodes[:8]
        nodes_dqs = np.sum(nodes_dq * weight_ex, axis=1).reshape(-1,1)
        nodes_dqh_norm = np.sqrt(np.sum(nodes_dqs*nodes_dqs,axis=0))
        nodes_dqh = nodes_dqs / nodes_dqh_norm
        
        dq = DualQuaternion(*nodes_dqh)
        
        x_ = frame.world2cam(dq.transform(x))
        x2 = frame.pix2cam(frame.cam2pix(x_))
        
        if np.isnan(x2[2]):
            return None
            
        n_ = dq.rotation.rotate(n)
        n_ = frame.world2cam(n, rotate_only=True)
        
        

        dweight = dist/(2*weight*weight)


        ddq = np.array([np.hstack([np.identity(8), nodes_dq[:,i].reshape(-1,1)*dweight[:,i]]) * weight_ex[:,i] for i in range(self.k)])
        
        ddqh = np.identity(8)/nodes_dqh_norm - np.dot(nodes_dqh,nodes_dqh.T) / np.power(nodes_dqh_norm, 3)
       
        dT1 = dq.d_transform(x)
        dT2 = np.hstack([dq.rotation.d_rotate(n), np.zeros((3,4))])
    
        
        cost = np.sum(n_*(x_-x2))
        j1 = np.dot(dT1, ddqh)
        j2 = np.dot(dT2, ddqh)

        jacobian = np.array([np.sum(np.dot(j1, ddq[i])*n_ + np.dot(j2,ddq[i])*(x_-x2), axis=0) for i in range(self.k)])
        
        return cost, jacobian

    def optimize(self, X):
        def cost(nodes):
            pass

        def jacobian(nodes):
            pass



if __name__ == "__main__":
    print(WarpField(100)._fit(np.array([0,0,1]), np.array([1,0,0]), [3,2,1], Frame(0)))
