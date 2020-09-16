from sklearn.neighbors import KDTree
import cv2 as cv
import numpy as np
import copy
import numbers
from frame import Frame

# references:
# https://github.com/mihaibujanca/dynamicfusion/blob/master/kfusion/src/utils/dual_quaternion.hpp
# https://github.com/mihaibujanca/dynamicfusion/blob/master/kfusion/src/utils/quaternion.hpp
# https://github.com/Poofjunior/QPose/blob/master/src/quaternion.hpp
# https://www.thinkmind.org/download.php?articleid=intsys_v6_n12_2013_5
class Quaternion:
    __array_priority__ = 15.0   # I need it to make rmul with np.arrays work

    def __init__(self):
        self.w = 1
        self.x = 0
        self.y = 0
        self.z = 0

    def encodeArray(self, x):
        self.w = x[0]
        self.x = x[1]
        self.y = x[2]
        self.z = x[3]

        return self

    def encodeAngleAxis(self, theta, normal):
        sin = np.sin(theta/2)
        normal = nomral / np.linalg.norm(normal) * sin
        
        self.w = np.cos(theta/2)
        self.x = normal[0]
        self.y = normal[1]
        self.z = normal[2]

        return self

    def conjugate(self):
        # w -x -y -z

        q = copy.deepcopy(self)
        q.x *= -1
        q.y *= -1
        q.z *= -1
        
        return q

    def rotate(self, x):
        return self*x*self.conjugate()

    def l2norm(self):
        return self.w*self.w + self.x*self.x + self.y*self.y + self.z*self.z

    def __add__(self, other):
        if isinstance(other, Quaternion):
            q = copy.deepcopy(self)
            q.w += other.w
            q.x += other.x
            q.y += other.y
            q.z += other.z

            return q
        else:
            raise TypeError

    def __sub__(self, other):
        if isinstance(other, Quaternion):
            q = copy.deepcopy(self)
            q.w -= other.w
            q.x -= other.x
            q.y -= other.y
            q.z -= other.z

            return q
        else:
            raise TypeError

    def __truediv__(self, other):
        if isinstance(other, numbers.Number):
            q = copy.deepcopy(self)
            q.w /= other
            q.x /= other
            q.y /= other
            q.z /= other

            return q
        else:
            raise TypeError

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            # self * other
            #
            # other         self
            # w -x -y -z  |  w
            # x  w  z -y  |  x
            # y -z  w  x  |  y
            # z  y -x  w  |  z
            # 
            q = copy.deepcopy(self)
            q.w = self.w*other.w - self.x*other.x - self.y*other.y - self.z*other.z
            q.x = self.w*other.x + self.x*other.w + self.y*other.z - self.z*other.y
            q.y = self.w*other.y - self.x*other.z + self.y*other.w + self.z*other.x
            q.z = self.w*other.z + self.x*other.y - self.y*other.x + self.z*other.w

            return q

        elif isinstance(other, (numbers.Number, float)):
            q = copy.deepcopy(self)
            q.w *= other
            q.x *= other
            q.y *= other
            q.z *= other

            return q

        elif isinstance(other, np.ndarray):
            # assume vertical array

            x = np.zeros(3)
            x[0] = self.w*other[0] + self.y*other[2] - self.z*other[1]
            x[1] = self.w*other[1] - self.x*other[2] + self.z*other[0]
            x[2] = self.w*other[2] + self.x*other[1] - self.y*other[0]

            return x

        else:
            raise TypeError

    def __rmul__(self, other):
        """
        if isinstance(other, Quaternion):
            # other * self

            q = copy.deepcopy(self)
            q.w = other.w*self.w - other.x*self.x - other.y*self.y - other.z*self.z
            q.x = other.w*self.x + other.x*self.w + other.y*self.z - other.z*self.y
            q.y = other.w*self.y - other.x*self.z + other.y*self.w + other.z*self.x
            q.z = other.w*self.z + other.x*self.y - other.y*self.x + other.z*self.w

            return q

        el"""
        if isinstance(other, numbers.Number):
            q = copy.deepcopy(self)
            q.w *= other
            q.x *= other
            q.y *= other
            q.z *= other

            return q

        elif isinstance(other, np.ndarray):
            # assume vertical array
            x = np.zeros(3)
            x[0] = other[0]*self.w + other[1]*self.z - other[2]*self.y
            x[1] = other[0]*self.z + other[1]*self.w + other[2]*self.x
            x[2] = other[0]*self.y - other[1]*self.x + other[2]*self.w

            return x

        else:
            raise TypeError

    def __getitem__(self, item):
        if isinstance(item, int):
            if item==0:
                return self.w
            elif item==1:
                return self.x
            elif item==2:
                return self.y
            elif item==3:
                return self.z
            else:
                raise IndexError
        elif isinstance(item, (list, tuple)):
            return [self[x] for x in item]
        elif isinstance(item, np.ndarray):
            return np.array([self[x] for x in item])
        elif isinstance(item, slice):
            return [self[i] for i in range(*item.indices(4))]
        else:
            raise TypeError

    def __str__(self):
        return f"{self.w} {self.x} {self.y} {self.z}"


class DualQuaternion:
    def __init__(self, rotation, translation):
        self.rotation = rotation
        self.translation = 0.5*translation*rotation

    def getTranslation(self):
        return np.array((2 * self.translation * self.rotation.conjugate())[1:])

    def transform(self, x):
        return self.rotation.rotate(x) + self.getTranslation()

    def l2norm(self):
        return self.rotation.l2norm() + self.translation.l2norm()

    def __add__(self, other):
        if isinstance(other, DualQuaternion):
            q = copy.deepcopy(self)
            q.rotation = q.rotation + other.rotation
            q.translation = q.translation + other.translation

            return q
        else:
            raise TypeError

    def __sub__(self, other):
        if isinstance(other, DualQuaternion):
            q = copy.deepcopy(self)
            q.rotation = q.rotation - other.rotation
            q.translation = q.translation - other.translation

            return q
        else:
            raise TypeError

    def __mul__(self, other):
        if isinstance(other, DualQuaternion):
            q = copy.deepcopy(self)
            q.rotation = self.rotation * other.rotation
            q.translation = self.rotation*other.translation + self.translation*other.rotation

            return q

        elif isinstance(other, numbers.Number):
            q = copy.deepcopy(self)
            q.rotation = other * q.rotation
            q.translation = other * q.translation

            return q

        else:
            raise TypeError

    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            q = copy.deepcopy(self)
            q.rotation = other * q.rotation
            q.translation = other * q.translation

            return q

        else:
            raise TypeError

    def __truediv__(self, other):
        if isinstance(other, numbers.Number):
            q = copy.deepcopy(self)
            q.rotation = q.rotation / other
            q.translation = q.translation / other

            return q

        elif isinstance(other, (list, tuple, np.array)):
            q = copy.deepcopy(self)
            q.rotation = q.rotation / other[0]
            q.translation = q.translation / other[1]

            return q

        else:
            raise TypeError

    def __getitem__(self, item):
        if isinstance(item, int):
            if item==0:
                return self.rotation.w
            elif item==1:
                return self.rotation.x
            elif item==2:
                return self.rotation.y
            elif item==3:
                return self.rotation.z
            elif item==4:
                return self.translation.w
            elif item==5:
                return self.translation.x
            elif item==6:
                return self.translation.y
            elif item==7:
                return self.translation.z
            else:
                raise IndexError
        elif isinstance(item, (list, tuple)):
            return [self[x] for x in item]
        elif isinstance(item, np.array):
            return np.array([self[x] for x in item])
        elif isinstance(item, slice):
            return [self[i] for i in range(*item.indices(8))]
        else:
            raise TypeError

    def __str__(self):
        return self.rotation.__str__() + " " + self.translation.__str__()


class WarpField:
    def __init__(self, frame1, frame2):
        self.frame1 = frame1    # original frame
        self.frame2 = frame2    # target frame to warp towards

        self.nodes_pos = self.init_nodes_with_pixel()[:,:1000]

        n = self.nodes_pos.shape[1]

        self.nodes_weight = np.ones(n).astype(np.double)
        self.nodes_dq = np.array([])

        self.kdtree = KDTree(self.nodes_pos.T)
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
        return self.frame1.cam2world(self.frame1.pixel_3d_cam())

    def knn(self, x, k=None):
        if k==None:
            k = self.k

        result = self.kdtree.query(x.reshape(-1,3), k=k)
        return result

    def getWarp(self, x, k=None):
        if k==None:
            k = self.k

        dist, ind = self.knn(x, k)
        dist = dist.reshape(-1).astype(np.double)
        ind = ind.reshape(-1)

        dq_list = self.nodes_dq[ind]
        dq_w = self.nodes_weight[ind]

        # weighted sum of quaternions
        w = np.exp(-dist*dist/(2*dq_w*dq_w))
        q = np.sum([w[i]*dq_list[i] for i in range(k)])
        

        # normalize
        q_norm = np.sqrt(q.l2norm())
        q = q/q_norm

        # return dq

        return q


    def residual1(self, frame, x, k=None):
        dq = self.getWarp(x, k)

        # n_ = dq.transform(n), x_ = dq.transform(x)

        # x_ -> project to frame (=x_nf), then get difference 

        # dot(n_, x_ - x_nf)

        # loss func

        pass

    def residual2(self, x):
        pass

if __name__ == "__main__":
    print(WarpField(Frame(0), Frame(1)).getWarp(np.array([0,0,0])))

    print(DualQuaternion(Quaternion().encodeArray([0,1,0,0]), Quaternion().encodeArray([0,0,0,0])).transform(np.array([1, 3, 0])))