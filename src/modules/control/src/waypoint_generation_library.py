#!/usr/bin/env python3
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

# TODO: add logic to shift desiredTimes for waypoint selection if acceleration limits are exceeded
class WaypointGen(object):
    def __init__(self):
        PI = 3.14159
        # NOTE: consider putting these entries into a YAML file and then read
        # in the order of [x_pos, y_pos, z_pos, yaw_angle] in [m, m, m, rad]
        # self.desiredPos = np.array([[0, 0, 0, 0],   
        #                             [1, 2, 5, 0],
        #                             [4, 5, 8, 0],
        #                             [3, 6, 10, 0],
        #                             [3, 6, 2, 0]])
        self.desiredPos = np.array([[0, 0, 0, 0],                                       
                                    [3, 6, 10, 0]])
        # in the order of [x_vel, y_vel, z_vel, yaw_rate] in [m/s, m/s, m/s, rad/s]
        self.desiredVel = np.array([[0, 0, 0, 0],
                                    [0, 0, 0, 0]])
        # in the order of [x_acc, y_acc, z_acc, yaw_acc] in [m/s^2, m/s^2, m/s^2, rad/s^2]
        self.desiredAcc = np.array([[0, 0, 0, 0],
                                    [0, 0, 0, 0]])
        # desired time to arrive at each waypoint
        # self.desiredTimes = np.array([0, 5, 10, 15, 25])
        self.desiredTimes = np.array([0, 10])
        # number of points between each waypoint, NOTE: should consider making this a function of sampling time with respect to the time difference between each desiredTimes entry
        self.numPtsBtTimes = 100        

    def lin_interpolation(self, desiredTimes, numPtsBtTimes):
        """ Linear interpolation between each point fed into the function with the given number of points in between"""
        timeVec = []
        # do the interpolation
        for i in range(np.size(desiredTimes)-1):
            for j in range(numPtsBtTimes):
                timeVec.append(desiredTimes[i] + (desiredTimes[i+1] - desiredTimes[i])*(j/numPtsBtTimes))
        # append the last point
        timeVec.append(desiredTimes[i+1])
        return timeVec

    def find_coeff_vector(self, coeffVectorList, currTime, timeVec):
        """ Function to find the coefficient vector to use for the waypoint generation and the index to shift for the time difference"""        
        # select the nearest index corresponding to the current time in the desiredTimes list
        nearestIdx = np.searchsorted(self.desiredTimes, currTime)
        if nearestIdx >= np.size(self.desiredTimes):
            nearestIdx = np.size(self.desiredTimes)-1
        elif nearestIdx == 0:
            nearestIdx = 1
        indexShift = np.where(timeVec == self.desiredTimes[nearestIdx-1])
        indexShift = int(indexShift[0])

        # find the shape of the desiredPos array (equivalent to size() in Matlab)
        arrayShape = np.shape(self.desiredPos)  

        # if there are only two points then there is no coeffVectorList
        if arrayShape[0] == 2:
            coeffVector = coeffVectorList
        else: 
            coeffVector = coeffVectorList[nearestIdx-1,:,:]
        return coeffVector, indexShift
    
    def gen_waypoints(self, coeffVectorList):
        """ Function to apply minimum jerk trajectory coefficients to calculate minimum jerk position, velocity, and acceleration"""
        timeVec = np.array(self.lin_interpolation(self.desiredTimes, self.numPtsBtTimes))
        waypoints = np.zeros((np.size(timeVec),4))
        desVel = np.zeros((np.size(timeVec),4))
        desAcc = np.zeros((np.size(timeVec),4))

        for i in range(0, 4):
            for m in range(0, np.size(timeVec)):
                # figure out which coeffVector to use
                coeffVectorApp, indexShift = self.find_coeff_vector(coeffVectorList, timeVec[m], timeVec)
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

    def waypoint_calculation_pt2pt(self):
        """ Calculate an optimal minimum jerk trajectory for two points"""
        for i in range(0, 4):
            # find the shape of the desiredPos array (equivalent to size() in Matlab)
            arrayShape = np.shape(self.desiredPos)  
            # temporary kinematics vector for two point formulation     
            tempKinematics = np.vstack((np.array([self.desiredPos[:,i]]).T, np.array([self.desiredVel[:,i]]).T, np.array([self.desiredAcc[:,i]]).T))
            # use loop to take out each element separately in tempKinematics
            desiredKinematics = np.zeros((np.size(tempKinematics), 1))    
            for k in range(0, np.size(tempKinematics)):
                desiredKinematics[k][0] = tempKinematics[k]
            timeDiff = self.desiredTimes[1] - self.desiredTimes[0]
            coeffMapMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                       [1, timeDiff, pow(timeDiff, 2), pow(timeDiff, 3), pow(timeDiff, 4), pow(timeDiff, 5)],
                                       [0, 1, 0, 0, 0, 0],
                                       [0, 1, 2*timeDiff, 3*pow(timeDiff, 2), 4*pow(timeDiff, 3), 5*pow(timeDiff, 4)],
                                       [0, 0, 2, 0, 0, 0],
                                       [0, 0, 2, 6*timeDiff, 12*pow(timeDiff, 2), 20*pow(timeDiff, 3)]])

            # perform the mapping from desired points to coefficients for sub-optimal minimum jerk waypoints
            if i == 0:
                coeffVector = np.dot(np.linalg.pinv(coeffMapMatrix), desiredKinematics)
            else:
                coeffVector = np.append(coeffVector, np.dot(np.linalg.pinv(coeffMapMatrix), desiredKinematics), axis = 1)    

        return self.gen_waypoints(coeffVector)

    def waypoint_calculation(self):
        """ Calculate a sub-optimal minimum jerk trajectory to get intermediate velocities and accelerations between the
            desired points and then compute optimal minimum jerk trajectory between each desired point"""
        # find the shape of the desiredPos array (equivalent to size() in Matlab)
        arrayShape = np.shape(self.desiredPos) 
        if arrayShape[0] == 2:
            return self.waypoint_calculation_pt2pt()
        else:
            for i in range(0, 4):
                # temporary kinematics vector for pseudoinverse formulation   
                tempKinematics = np.vstack((np.array([self.desiredPos[:,i]]).T, np.array([self.desiredVel[:,i]]).T, np.array([self.desiredAcc[:,i]]).T))
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

            # calculate the velocity and acceleration at each of the desired waypoints
            pinvDesiredVel = np.zeros((arrayShape[0]-2,4))
            pinvDesiredAcc = np.zeros((arrayShape[0]-2,4))
            for k in range(0, arrayShape[0]-2):
                for i in range(0, 4):
                    # velocity waypoints
                    for j in range(1, 6):
                        if j == 1:
                            pinvDesiredVel[k][i] = coeffVector[j][i]
                        else:
                            # don't need extra variable for multiplication factor when taking derivative of the position waypoints equation can just use j
                            pinvDesiredVel[k][i] = pinvDesiredVel[k][i] + j*coeffVector[j][i]*np.power(self.desiredTimes[k+1], j-1)
                    # acceleration waypoints
                    for j in range(2, 6):
                        if j == 2:
                            pinvDesiredAcc[k][i] = 2*coeffVector[j][i]
                        else:
                            # taking derivative of velocity waypoints equation for desired acceleration
                            if j == 3:
                                multFactor = 6
                            elif j == 4:
                                multFactor = 12
                            elif j == 5:
                                multFactor = 20
                            pinvDesiredAcc[k][i] = pinvDesiredAcc[k][i] + multFactor*coeffVector[j][i]*np.power(self.desiredTimes[k+1], j-2)
            for k in range(0, arrayShape[0]-1):
                # take the desired vel and accel from each waypoint and use as initial conditions for optimal min jerk trajectory      
                minDesiredPos = np.array([self.desiredPos[k,:],
                                        self.desiredPos[k+1,:]])
                if k == 0:
                    minDesiredVel = np.vstack((self.desiredVel[0,:], pinvDesiredVel[k,:]))
                    minDesiredAcc = np.vstack((self.desiredAcc[0,:], pinvDesiredAcc[k,:]))
                elif k == (arrayShape[0]-2):
                    minDesiredVel = np.vstack((pinvDesiredVel[k-1,:], self.desiredVel[-1,:]))
                    minDesiredAcc = np.vstack((pinvDesiredAcc[k-1,:], self.desiredAcc[-1,:]))
                else:
                    minDesiredVel = np.vstack((pinvDesiredVel[k-1,:], pinvDesiredVel[k,:]))
                    minDesiredAcc = np.vstack((pinvDesiredAcc[k-1,:], pinvDesiredAcc[k,:]))
                timeDiff = self.desiredTimes[k+1] - self.desiredTimes[k]
                # min jerk coefficients mapping matrix
                if k == 0:
                    lastPtsCoeffMapMatrix = np.array([[[1, 0, 0, 0, 0, 0],
                                                    [1, timeDiff, pow(timeDiff, 2), pow(timeDiff, 3), pow(timeDiff, 4), pow(timeDiff, 5)],
                                                    [0, 1, 0, 0, 0, 0],
                                                    [0, 1, 2*timeDiff, 3*pow(timeDiff, 2), 4*pow(timeDiff, 3), 5*pow(timeDiff, 4)],
                                                    [0, 0, 2, 0, 0, 0],
                                                    [0, 0, 2, 6*timeDiff, 12*pow(timeDiff, 2), 20*pow(timeDiff, 3)]]])
                else:
                    lastPtsCoeffMapMatrix = np.append(lastPtsCoeffMapMatrix, np.array([[[1, 0, 0, 0, 0, 0],
                                                                                        [1, timeDiff, pow(timeDiff, 2), pow(timeDiff, 3), pow(timeDiff, 4), pow(timeDiff, 5)],
                                                                                        [0, 1, 0, 0, 0, 0],
                                                                                        [0, 1, 2*timeDiff, 3*pow(timeDiff, 2), 4*pow(timeDiff, 3), 5*pow(timeDiff, 4)],
                                                                                        [0, 0, 2, 0, 0, 0],
                                                                                        [0, 0, 2, 6*timeDiff, 12*pow(timeDiff, 2), 20*pow(timeDiff, 3)]]]), axis=0)
                # for x, y, z, and yaw positions
                for n in range(0, 4):
                    desiredKinematics = np.vstack((np.array([minDesiredPos[:,n]]).T, np.array([minDesiredVel[:,n]]).T, np.array([minDesiredAcc[:,n]]).T))
                    # perform the mapping from desired points to coefficients for optimal minimum jerk waypoints on last two points
                    if n == 0:
                        coeffVector2 = np.dot(np.linalg.inv(lastPtsCoeffMapMatrix[k,:,:]), desiredKinematics)
                    else:
                        coeffVector2 = np.append(coeffVector2, np.dot(np.linalg.inv(lastPtsCoeffMapMatrix[k,:,:]), desiredKinematics), axis = 1) 
                if k == 0:
                    # turn coeffVector2 into 3D for storage
                    coeffVectorList = np.array([coeffVector2])
                else:
                    coeffVectorList = np.append(coeffVectorList, np.array([coeffVector2]), axis=0)

            return self.gen_waypoints(coeffVectorList)
