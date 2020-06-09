#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
import rospy
import rosbag
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from mpl_toolkits import mplot3d

# class WaypointCoefficients(object):
#     def __init__(self):
#         self.waypointPublisher = rospy.Publisher("/control/waypoint_coefficients", , queue_size = 1)

PI = 3.14159
# in the order of [x_pos, y_pos, z_pos, yaw_angle] in [m, m, m, rad]
desiredPos = np.array([[0, 0, 0, 0],   
                       [1, 2, 5, 0],
                       [3, 6, 10, 0]])

desiredVel = np.array([[0, 0, 0, 0],
                       [0, 0, 0, 0]])

desiredAcc = np.array([[0, 0, 0, 0],
                       [0, 0, 0, 0]])

desiredTimes = np.array([0, 10, 20])

for i in range(0, 4):
    # find the shape of the desiredPos array (equivalent to size() in Matlab)
    arrayShape = np.shape(desiredPos)
    # # need to use loop to take out each element separately
    # for k in range(0, arrayShape[0]):
    #     if k == 0:
    #         desiredKinematics = 
    # desiredKinematics = np.array([[desiredPos[:,i]],
    #                               [desiredVel[:,i]],
    #                               [desiredAcc[:,i]]])
    tempKinematics = desiredPos[:,i]
    tempKinematics = np.append(tempKinematics, desiredVel[:,i], axis = 0)
    tempKinematics = np.append(tempKinematics, desiredAcc[:,i], axis = 0)

    desiredKinematics = np.zeros((np.size(tempKinematics), 1))
    print(desiredKinematics)
    for k in range(0, np.size(tempKinematics)):
        desiredKinematics[k][0] = tempKinematics[k]

    print(desiredKinematics)    
    # just the coefficients mapping the initial and final desired positions for now 
    # TODO: Fix the incorrect mapping for positions
    coeffMapMatrix = np.array([[1, 0, 0, 0, 0, 0],
                               [1, desiredTimes[-1], pow(desiredTimes[-1], 2), pow(desiredTimes[-1], 3), pow(desiredTimes[-1], 4), pow(desiredTimes[-1], 5)]])
    if arrayShape[0] > 2:
        for j in range(1, np.size(desiredTimes)-1):
            temp = [[1, desiredTimes[j], pow(desiredTimes[j], 2), pow(desiredTimes[j], 3), pow(desiredTimes[j], 4), pow(desiredTimes[j], 5)]]
            # print(temp)
            # print(coeffMapMatrix)
            coeffMapMatrix = np.append(coeffMapMatrix, temp, axis = 0)

            # print(coeffMapMatrix)
    
    # add the velocity and acceleration terms
    temp2 = [[0, 1, 0, 0, 0, 0],
             [0, 1, 2*desiredTimes[-1], 3*pow(desiredTimes[-1], 2), 4*pow(desiredTimes[-1], 3), 5*pow(desiredTimes[-1], 4)],
             [0, 0, 2, 0, 0, 0],
             [0, 0, 2, 6*desiredTimes[-1], 12*pow(desiredTimes[-1], 2), 20*pow(desiredTimes[-1], 3)]]
    # print(temp2)
    coeffMapMatrix = np.append(coeffMapMatrix, temp2, axis = 0)
    # print(coeffMapMatrix)
    if i == 0:
        coeffVector = np.dot(np.linalg.pinv(coeffMapMatrix), desiredKinematics)
    else:
        coeffVector = np.append(coeffVector, np.dot(np.linalg.pinv(coeffMapMatrix), desiredKinematics), axis = 1)
    
    print(coeffVector)

fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot3D(desiredPos[:,0], desiredPos[:,1], desiredPos[:,2], 'ro')
# create time vector
timeVec = np.linspace(desiredTimes[0], desiredTimes[-1], num = 200)
waypoints = np.zeros((200,4))
for i in range(0, 4):
    for j in range(0, 6):
        if j == 0:
            waypoints[:,i] = coeffVector[j][i]
        else:
            waypoints[:,i] = waypoints[:,i] + coeffVector[j][i]*np.power(timeVec, j)

    # if i == 0:
    #     # xPosWaypoints = coeffVector[0][i] + coeffVector[1][i]*timeVec + coeffVector[2][i]*np.power(timeVec, 2) + coeffVector[3][i]*np.power(timeVec, 3) + 

    # elif i == 1:
    # elif i == 2:
    # elif i == 3:
print(np.shape(waypoints))
ax.plot3D(waypoints[:,0], waypoints[:,1], waypoints[:,2])
print(waypoints)
plt.show()


            