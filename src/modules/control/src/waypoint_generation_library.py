#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
import rospy
import rosbag
import numpy as np
from scipy import linalg
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion, quaternion_from_euler

# TODO: calculate maximum accel limits and then shift the desired times by an appropriate amount to not exceed accel constraints
class WaypointGen(object):
    def __init__(self):
        PI = 3.14159
        # NOTE: consider putting these entries into a YAML file and then read
        # in the order of [x_pos, y_pos, z_pos, yaw_angle] in [m, m, m, rad]
        # self.desiredPos = np.array([[0, 0, 0, 0],   
        #                             [1, 2, 5, 0],
        #                             [4, 5, 8, 0],
        #                             [3, 6, 10, 0]])
        self.desiredPos = np.array([[0, 0, 0, 0],                                       
                                    [2, 3, 5, 0]])
        # in the order of [x_vel, y_vel, z_vel, yaw_rate] in [m/s, m/s, m/s, rad/s]
        self.desiredVel = np.array([[0, 0, 0, 0],
                                    [0, 0, 0, 0]])
        # in the order of [x_acc, y_acc, z_acc, yaw_acc] in [m/s^2, m/s^2, m/s^2, rad/s^2]
        self.desiredAcc = np.array([[0, 0, 0, 0],
                                    [0, 0, 0, 0]])
        # desired time to arrive at each waypoint
        # self.desiredTimes = np.array([0, 5, 10, 15])
        self.desiredTimes = np.array([0, 10])
        # number of points between each waypoint, should consider making this a function of sampling time with respect to the time difference between each desiredTimes entry
        self.numPtsBtTimes = 100        

    def lin_interpolation(self, desiredTimes, numPtsBtTimes):
        """Linear interpolation between each point fed into the function with the given number of points in between"""
        timeVec = []
        # do the interpolation
        for i in range(np.size(desiredTimes)-1):
            for j in range(numPtsBtTimes):
                timeVec.append(desiredTimes[i] + (desiredTimes[i+1] - desiredTimes[i])*(j/numPtsBtTimes))
        # append the last point
        timeVec.append(desiredTimes[i+1])
        return timeVec

    def waypoint_calculation(self):
        """Calculate a sub-optimal minimum jerk trajectory and replace the trajectory between the second to last point and the
        last point with an optimal minimum jerk trajectory"""
        for i in range(0, 4):
            # find the shape of the desiredPos array (equivalent to size() in Matlab)
            arrayShape = np.shape(self.desiredPos)    
            tempKinematics = self.desiredPos[:,i]
            tempKinematics = np.append(tempKinematics, self.desiredVel[:,i], axis = 0)
            tempKinematics = np.append(tempKinematics, self.desiredAcc[:,i], axis = 0)
            # use loop to take out each element separately in tempKinematics
            desiredKinematics = np.zeros((np.size(tempKinematics), 1))    
            for k in range(0, np.size(tempKinematics)):
                desiredKinematics[k][0] = tempKinematics[k]

            # just the mapping coefficients mapping the initial position
            coeffMapMatrix = np.array([[1, 0, 0, 0, 0, 0]])                               
            # insert the mapping coefficients for intermediate points
            if arrayShape[0] > 2:
                for j in range(1, np.size(self.desiredTimes)-1):
                    temp = [[1, self.desiredTimes[j], pow(self.desiredTimes[j], 2), pow(self.desiredTimes[j], 3), pow(self.desiredTimes[j], 4), pow(self.desiredTimes[j], 5)]]
                    coeffMapMatrix = np.append(coeffMapMatrix, temp, axis = 0)
            # add the final desired position
            coeffMapMatrix = np.append(coeffMapMatrix, [[1, self.desiredTimes[-1], pow(self.desiredTimes[-1], 2), pow(self.desiredTimes[-1], 3), pow(self.desiredTimes[-1], 4), pow(self.desiredTimes[-1], 5)]], axis = 0)
            # add the velocity and acceleration terms
            temp2 = [[0, 1, 0, 0, 0, 0],
                    [0, 1, 2*self.desiredTimes[-1], 3*pow(self.desiredTimes[-1], 2), 4*pow(self.desiredTimes[-1], 3), 5*pow(self.desiredTimes[-1], 4)],
                    [0, 0, 2, 0, 0, 0],
                    [0, 0, 2, 6*self.desiredTimes[-1], 12*pow(self.desiredTimes[-1], 2), 20*pow(self.desiredTimes[-1], 3)]]
            coeffMapMatrix = np.append(coeffMapMatrix, temp2, axis = 0)

            # perform the mapping from desired points to coefficients for sub-optimal minimum jerk waypoints
            if i == 0:
                coeffVector = np.dot(np.linalg.pinv(coeffMapMatrix), desiredKinematics)
            else:
                coeffVector = np.append(coeffVector, np.dot(np.linalg.pinv(coeffMapMatrix), desiredKinematics), axis = 1)    

        # create time vector
        timeVec = np.array(self.lin_interpolation(self.desiredTimes, self.numPtsBtTimes))
        waypoints = np.zeros((np.size(timeVec),4))
        desVel = np.zeros((np.size(timeVec),4))
        desAcc = np.zeros((np.size(timeVec),4))

        # calculate the velocity and acceleration for the second to last waypoint
        secondLastVel = np.zeros((1,4))
        secondLastAcc = np.zeros((1,4))
        for i in range(0, 4):
            # velocity waypoints
            for j in range(1, 6):
                if j == 1:
                    secondLastVel[0][i] = coeffVector[j][i]
                else:
                    # don't need extra variable for multiplication factor when taking derivative of the position waypoints equation can just use j
                    secondLastVel[0][i] = secondLastVel[0][i] + j*coeffVector[j][i]*np.power(self.desiredTimes[np.size(self.desiredTimes)-2], j-1)
            # acceleration waypoints
            for j in range(2, 6):
                if j == 2:
                    secondLastAcc[0][i] = 2*coeffVector[j][i]
                else:
                    # taking derivative of velocity waypoints equation for desired acceleration
                    if j == 3:
                        multFactor = 6
                    elif j == 4:
                        multFactor = 12
                    elif j == 5:
                        multFactor = 20
                    secondLastAcc[0][i] = secondLastAcc[0][i] + multFactor*coeffVector[j][i]*np.power(self.desiredTimes[np.size(self.desiredTimes)-2], j-2)
        
        # take the desired vel and accel from the 2nd to last waypoint and use as initial conditions for optimal min jerk trajectory
        # 2nd to last point index        
        minDesiredPos = np.array([self.desiredPos[np.size(self.desiredTimes)-2,:],
                                self.desiredPos[np.size(self.desiredTimes)-1,:]])
        minDesiredVel = np.concatenate((secondLastVel, np.array([[self.desiredVel[1,0], self.desiredVel[1,1], self.desiredVel[1,2], self.desiredVel[1,3]]])), axis = 0)
        minDesiredAcc = np.concatenate((secondLastAcc, np.array([[self.desiredAcc[1,0], self.desiredAcc[1,1], self.desiredAcc[1,2], self.desiredAcc[1,3]]])), axis = 0)
        timeDiff = self.desiredTimes[-1] - self.desiredTimes[np.size(self.desiredTimes)-2]
        # min jerk coefficients mapping matrix
        lastPtsCoeffMapMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                        [1, timeDiff, pow(timeDiff, 2), pow(timeDiff, 3), pow(timeDiff, 4), pow(timeDiff, 5)],
                                        [0, 1, 0, 0, 0, 0],
                                        [0, 1, 2*timeDiff, 3*pow(timeDiff, 2), 4*pow(timeDiff, 3), 5*pow(timeDiff, 4)],
                                        [0, 0, 2, 0, 0, 0],
                                        [0, 0, 2, 6*timeDiff, 12*pow(timeDiff, 2), 20*pow(timeDiff, 3)]])
        # calculate the position/velocity/acceleration target trajectory between the waypoints
        for n in range(0, 4):
            # python is being dumb
            desiredKinematics = np.array([minDesiredPos[:,n]]).T
            desiredKinematics = np.append(desiredKinematics, np.array([minDesiredVel[:,n]]).T, axis = 0)
            desiredKinematics = np.append(desiredKinematics, np.array([minDesiredAcc[:,n]]).T, axis = 0)
            # perform the mapping from desired points to coefficients for optimal minimum jerk waypoints on last two points
            if n == 0:
                coeffVector2 = np.dot(np.linalg.inv(lastPtsCoeffMapMatrix), desiredKinematics)
            else:
                coeffVector2 = np.append(coeffVector2, np.dot(np.linalg.inv(lastPtsCoeffMapMatrix), desiredKinematics), axis = 1) 
        for i in range(0, 4):
            for m in range(0, np.size(timeVec)):
                if timeVec[m] <= self.desiredTimes[np.size(self.desiredTimes)-2]:
                    coeffVectorApp = coeffVector
                    indexShift = 0
                else:
                    coeffVectorApp = coeffVector2
                    indexShift = np.where(timeVec == self.desiredTimes[np.size(self.desiredTimes)-2])
                    indexShift = int(indexShift[0])
                # position waypoints
                for j in range(0, 6):
                    if j == 0:
                        waypoints[m,i] = coeffVectorApp[j][i]
                    else:                    
                        waypoints[m,i] = waypoints[m,i] + coeffVectorApp[j][i]*pow(timeVec[m] - timeVec[indexShift], j)
                # velocity waypoints
                for j in range(1, 6):
                    if j == 1:
                        desVel[m,i] = coeffVectorApp[j][i]
                    else:
                        # don't need extra variable for multiplication factor when taking derivative of the position waypoints equation can just use j
                        desVel[m,i] = desVel[m,i] + j*coeffVectorApp[j][i]*pow(timeVec[m] - timeVec[indexShift], j-1)
                # acceleration waypoints
                for j in range(2, 6):
                    if j == 2:
                        desAcc[m,i] = 2*coeffVectorApp[j][i]
                    else:
                        # taking derivative of velocity waypoints equation for desired acceleration
                        if j == 3:
                            multFactor = 6
                        elif j == 4:
                            multFactor = 12
                        elif j == 5:
                            multFactor = 20
                        desAcc[m,i] = desAcc[m,i] + multFactor*coeffVectorApp[j][i]*pow(timeVec[m] - timeVec[indexShift], j-2)
        return waypoints, desVel, desAcc, timeVec
