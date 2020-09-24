import numpy as np
import copy
import numbers
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

    def array(self):
        return np.array([self.w, self.x, self.y, self.z])

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
        # assume (3,n) shape
        x = x.reshape(3,-1)
        x = np.vstack([np.zeros(x.shape[1]), x])

        return (self*x*self.conjugate())[1:]

    def d_rotate(self, x):
        a,b,c = x[0], x[1], x[2]
        w,x,y,z = self.w, self.x, self.y, self.z

        return 2*np.array([
            [a*w+b*z-c*y, a*x+b*y+c*z, -a*y+b*x-c*w, -a*z-b*w+c*x],
            [-a*z+b*w+c*x, a*y-b*x+c*w, a*x+b*y+c*z, -a*w-b*z+c*y],
            [a*y-b*x+c*w, a*z-b*w-c*x, a*w+b*z-c*y, a*x+b*y+c*z]]).reshape(3,4)

    """
    def d_rotate2(self):
        w,x,y,z = self.w, self.x, self.y, self.z

        return 2*np.array([[
            [w,x,-y,-z],
            [z,y,x,w],
            [-y,z,-w,z]],

            [[-z,y,x,-w],
            [w,-x,y,-z],
            [x,w,z,y]],

            [[y,z,w,x],
            [-x,-w,z,y],
            [w,-x,-y,z]
            ]])
    """
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
        elif isinstance(other, np.ndarray):
            q = copy.deepcopy(self)
            q.w += other[0]
            q.x += other[1]
            q.y += other[2]
            q.z += other[3]

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
        elif isinstance(other, np.ndarray):
            q = copy.deepcopy(self)
            q.w -= other[0]
            q.x -= other[1]
            q.y -= other[2]
            q.z -= other[3]

            return q
        else:
            raise TypeError(type(other))

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
            # other         self        or      self          other
            # w -x -y -z  |  w                  w -x -y -z  |  w
            # x  w  z -y  |  x                  x  w -z  w  |  x
            # y -z  w  x  |  y                  y  z  w -x  |  y
            # z  y -x  w  |  z                  z -y  x  w  |  z
            # 
            q = copy.deepcopy(self)
            q.w = self.w*other.w - self.x*other.x - self.y*other.y - self.z*other.z
            q.x = self.w*other.x + self.x*other.w + self.y*other.z - self.z*other.y
            q.y = self.w*other.y - self.x*other.z + self.y*other.w + self.z*other.x
            q.z = self.w*other.z + self.x*other.y - self.y*other.x + self.z*other.w

            return q

        elif isinstance(other, np.ndarray):

            q = np.zeros(other.shape)
            q[0] = self.w*other[0] - self.x*other[1] - self.y*other[2] - self.z*other[3]
            q[1] = self.w*other[1] + self.x*other[0] + self.y*other[3] - self.z*other[2]
            q[2] = self.w*other[2] - self.x*other[3] + self.y*other[0] + self.z*other[1]
            q[3] = self.w*other[3] + self.x*other[2] - self.y*other[1] + self.z*other[0]


            return q

        elif isinstance(other, (numbers.Number, float)):
            q = copy.deepcopy(self)
            q.w *= other
            q.x *= other
            q.y *= other
            q.z *= other

            return q

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
        """
        if isinstance(other, np.ndarray):

            q = np.zeros(other.shape)
            q[0] = other[0]*self.w - other[1]*self.x - other[2]*self.y - other[3]*self.z
            q[1] = other[0]*self.x + other[1]*self.w + other[2]*self.z - other[3]*self.y
            q[2] = other[0]*self.y - other[1]*self.z + other[2]*self.w + other[3]*self.x
            q[3] = other[0]*self.z + other[1]*self.y - other[2]*self.x + other[3]*self.w

            return q

        elif isinstance(other, numbers.Number):
            q = copy.deepcopy(self)
            q.w *= other
            q.x *= other
            q.y *= other
            q.z *= other

            return q

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
        # this is equivalent of (rot  0.5*trs*rot)*(1 0 0 0   0 x y z)
        # (rot   rot*(0 x y z)+0.5*trs*rot)
        return self.rotation.rotate(x) + self.getTranslation().reshape(3,-1)

    def d_transform(self, x):
        a = x

        w,x,y,z = self.rotation.w, self.rotation.x, self.rotation.y, self.rotation.z
        dtrans1_dtrans2 = 2*np.array([[x, w, z, -y], [y, -z, w, x], [z, y, -x, w]])

        return np.hstack([self.rotation.d_rotate(a), dtrans1_dtrans2])

    def l2norm(self):
        return self.rotation.l2norm() + self.translation.l2norm()

    def array(self):
        return np.array([self.rotation.w, self.rotation.x, self.rotation.y, self.rotation.z, 
        self.translation.w, self.translation.x, self.translation.y, self.translation.z])

    def __add__(self, other):
        if isinstance(other, DualQuaternion):
            q = copy.deepcopy(self)
            q.rotation = q.rotation + other.rotation
            q.translation = q.translation + other.translation

            return q
        elif isinstance(other, np.ndarray):
            q = copy.deepcopy(self)
            q.rotation = q.rotation + other[:4]
            q.translation = q.translation + other[4:]

            return q
        else:
            raise TypeError

    def __sub__(self, other):
        if isinstance(other, DualQuaternion):
            q = copy.deepcopy(self)
            q.rotation = q.rotation - other.rotation
            q.translation = q.translation - other.translation

            return q
        elif isinstance(other, np.ndarray):
            q = copy.deepcopy(self)
            q.rotation = q.rotation - other[:4]
            q.translation = q.translation - other[4:]

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
        if isinstance(item, (int, np.int32)):
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
        elif isinstance(item, np.ndarray):
            return np.array([self[x] for x in item])
        elif isinstance(item, slice):
            return [self[i] for i in range(*item.indices(8))]
        else:
            raise TypeError

    def __str__(self):
        return self.rotation.__str__() + " " + self.translation.__str__()