from sklearn.neighbors import KDTree
import cv2 as cv
import numpy as np
from scipy.sparse import coo_matrix, bmat, hstack, bsr_matrix

from frame import Frame
from quaternion import Quaternion, DualQuaternion


class WarpField:
    def __init__(self, frame1, frame2):
        self.frame1 = frame1    # original frame
        self.frame2 = frame2    # target frame to warp towards

        self.nodes_pos = self.init_nodes_with_pixel()

        n = self.nodes_pos.shape[1]

        self.nodes_weight = np.ones(n).astype(np.double)
        self.nodes_dq = np.array([])

        self.kdtree = KDTree(self.nodes_pos.T, 500)
        self.k = 3

        self.pose = np.dot(self.frame2.pose,self.frame1.pose_inv)

        for i in range(n):
            dq = DualQuaternion(Quaternion(), Quaternion().encodeArray([0,0,0,0]))
            self.nodes_dq = np.append(self.nodes_dq,dq)


    def init_nodes_with_SIFT(self):
        img_g = cv.cvtColor(self.frame1.img_rgb, cv.COLOR_RGB2GRAY)
        sift = cv.xfeatures2d.SIFT_create()
        kp = sift.detect(img_g, None)

        #print(kp)

        return self.frame1.cam2world(self.frame1.pix2cam(np.round(kp)))

    def init_nodes_with_pixel(self):
        h,w = self.frame1.h, self.frame1.w
        h_d, w_d = 10, 10
        m = int(w/w_d)
        n = int(h/h_d)*int(w/w_d)

        x = np.array([[i%m, i*m] for i in range(n)]).T

        return self.frame1.cam2world(self.frame1.pix2cam(x))


    def knn(self, X, k=None):
        # input should be (3,n)

        if k==None:
            k = self.k

        dist, ind = self.kdtree.query(X.T, k=k)
        dist, ind = dist.T, ind.T
        return dist, ind

    def getWarp(self, X, k=None, differential = False):
        # input should be (3,n)
        X = X.reshape(3,-1)

        if k==None:
            k = self.k

        dist, ind = self.knn(X, k)
        dq_list = self.nodes_dq[ind]
        dq_weight = self.nodes_weight[ind]

        # convert to (n,3)
        #X = X.T
        n = X.shape[1]

        # weighted sum of quaternions
        w = np.exp(-dist*dist/(2*dq_weight*dq_weight))
        q = np.sum(w*dq_list, axis=0)

        # normalize
        q_norm2 = np.array([q_.l2norm() for q_ in q])
        q_norm = np.sqrt(q_norm2)
        qh = q/q_norm


        if differential:
            range_np = np.array(range(8))
            q_arr = np.array([q_[range_np] for q_ in q]).reshape(-1,8,1)

            qqT = np.array([np.dot(q_arr_, q_arr_.T) for q_arr_ in q_arr])

            q_norm2 = q_norm2.reshape(-1,1,1)
            q_norm = q_norm.reshape(-1,1,1)

            dqh_dq = np.array([np.identity(8) for i in range(n)])/q_norm - qqT/q_norm/q_norm2

            return qh, dqh_dq, ind, dq_list, w
            
        else:
            return qh


    def residual1(self, X, k=None):
        # qh = q hat. normalized q
        # q = weighted sum of dualquaternions
        # qi = dualquaternion of a node
        # wi = weight of a node

        # input should be (3,n) in world coordinate
        X = X.reshape(3,-1)

        if k==None:
            k = self.k

        qh, dqh_dq, ind, dq_list, dq_w = self.getWarp(X, k, differential = True)
        nodes_n = self.nodes_pos.shape[1]

        frame1 = self.frame1
        frame2 = self.frame2

        n = X.shape[1]

        # compute normals
        N = frame1.normal(X)

        # back-project X with frame2 coordinates
        X_proj = frame2.pix2cam(frame2.cam2pix(X))

        # n_ = qh.transform(n). ignore transform
        N_ = np.array([qh[i].rotation.rotate(N[:,i]) for i in range(n)]).reshape(3,-1)
        dn_dq = np.array([np.hstack([np.dot(qh[i].rotation.d_rotate(N[:,i]), dqh_dq[i, :4, :4]), np.zeros((3,4))]) for i in range(n)])

        # x_ = qh.transform(x)
        X_ = frame2.world2cam(np.array([qh[i].transform(X[:,i]) for i in range(n)]).reshape(3,-1))
        dx_dq = np.array([np.dot(qh[i].d_transform(X[:,i]), dqh_dq[i, :8, :8]) for i in range(n)])


        # only consider valid pixels
        valid_pix = ~np.isnan(np.sum(N_, axis=0))
        n = np.sum(valid_pix)

        ind = ind[:, valid_pix]
        X_ = X_[:, valid_pix]
        X_proj = X_proj[:, valid_pix]
        N_ = N_[:, valid_pix]


        # loss func
        loss = np.sum(N_ * (X_ - X_proj), axis=0)

        print(np.mean(loss))


        # dq/dqi
        # TODO: implment dq_dwi
        # TODO: implement tukey error

        t = np.zeros((n, 9*k))

        rows = np.zeros(n*9*k).astype(int)
        cols = np.zeros(n*9*k).astype(int)

        for i in range(n):
            rows[i*9*k: (i+1)*9*k] = np.ones(9*k)*i
            cols[i*9*k: (i+1)*9*k] = np.concatenate([range(ind[j,i]*9, ind[j,i]*9+9) for j in range(k)])

            for j in range(k):
                # de/dq
                de_dq = np.dot((X_ - X_proj).T, dn_dq[i]) + np.dot(N_.T, dx_dq[i])
                dq_dqi = np.hstack([np.identity(8)*dq_w[j,i], dq_list[j,i].array().reshape(-1,1)])
                de_dqi = np.dot(de_dq[i], dq_dqi).reshape(1,9)

                if not np.isnan(np.sum(de_dqi)):
                    t[i,j*9:j*9+9] = de_dqi

        t = t.reshape(-1)

        jacobian = bsr_matrix((t,(rows,cols)), shape=(n, nodes_n*9), dtype=np.float64)
        hessian = jacobian.T.dot(jacobian)

        print(jacobian.mean())

        return loss, jacobian, hessian


    def residual2(self, x):
        pass

from scipy.integrate import quad

if __name__ == "__main__":
    frame0 = Frame(0)
    print(WarpField(frame0, Frame(100)).residual1(frame0.cam2world(frame0.pixel_3d_cam()[:,:5000])))

    dq = DualQuaternion(Quaternion().encodeArray([1,0,0,0]), Quaternion().encodeArray([0,0,0,0]))
    x = np.array([2,1,3]).reshape(3,-1)
    y = dq.transform(x)