import numpy as np

class QuatMath(object):
    """ Quaternion Math Functions"""
    
    def euler_to_quaternion(self, r, p, y):
            """ Function to convert euler angles to quaternion"""
            # assemble direction cosine matrix corresponding to 1-2-3 rotation
            a_DCM_b = np.array(([np.cos(y)*np.cos(p), np.cos(y)*np.sin(p)*np.sin(r) - np.sin(y)*np.cos(r), np.cos(y)*np.sin(p)*np.cos(r) + np.sin(r)*np.sin(y)],
                                [np.sin(y)*np.cos(p), np.sin(y)*np.sin(p)*np.sin(r) + np.cos(r)*np.cos(y), np.sin(y)*np.sin(p)*np.cos(r) - np.cos(y)*np.sin(r)],
                                [(-1)*np.sin(p), np.cos(p)*np.sin(r), np.cos(p)*np.cos(r)]))
            qw = 0.5*pow(1 + a_DCM_b[0,0] + a_DCM_b[1,1] + a_DCM_b[2,2], 0.5)
            qx = (a_DCM_b[2,1] - a_DCM_b[1,2])/(4*qw)
            qy = (a_DCM_b[0,2] - a_DCM_b[2,0])/(4*qw)
            qz = (a_DCM_b[1,0] - a_DCM_b[0,1])/(4*qw)
            a_q_b = np.array(([qx],
                            [qy],
                            [qz],
                            [qw]))
            return a_q_b

    def quat_inverse(self, a_q_b):
        """ Function to calculate inverse of quaternion"""
        b_q_a = (-1)*a_q_b
        # scalar term is always positive with current convention
        b_q_a[3,0] = a_q_b[3,0]
        return b_q_a

    def quat_mult(self, a_q_b, b_q_c):
        """ Function to multiply two quaternions together"""
        # intermediate matrix for matrix multiplication
        intermediateMat = np.array(([a_q_b[3,0], (-1)*a_q_b[2,0], a_q_b[1,0], a_q_b[0,0]],
                                    [a_q_b[2,0], a_q_b[3,0], (-1)*a_q_b[0,0], a_q_b[1,0]],
                                    [(-1)*a_q_b[1,0], a_q_b[0,0], a_q_b[3,0], a_q_b[2,0]],
                                    [(-1)*a_q_b[0,0], (-1)*a_q_b[1,0], (-1)*a_q_b[2,0], a_q_b[3,0]]))
        a_q_c = np.dot(intermediateMat, b_q_c)
        return a_q_c

    def calc_qw_check_norm(self, qx, qy, qz):
            """ Calculate scalar component of quaternion and check norm and normalize if norm is greater than one"""
            # turn warnings into errors
            warnings.filterwarnings("error")
            try:
                qw = pow(1 - pow(qx,2) - pow(qy,2) - pow(qz,2), 0.5)
            except:
                qw = 0
                print('sqrt of negative value attempted')
            quat = np.array(([float(qx), float(qy), float(qz), float(qw)]))
            # check the norm
            if np.linalg.norm(quat) > 1:
                quat = quat / np.linalg.norm(quat)
            return quat[0], quat[1], quat[2], quat[3]
