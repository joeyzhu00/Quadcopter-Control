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

PI = 3.14159
# in the order of [x_pos, y_pos, z_pos, yaw_angle] in [m, m, m, rad]
desiredPos = np.array([[0, 0, 0, 0],   
                        [1, 2, 5, 0],
                        [4, 5, 8, 0],
                        [3, 6, 10, PI/2]])

# desiredPos = np.array([[0, 0, 0, 0],   
#                     [3, 6, 10, PI/2]])

desiredVel = np.array([[0, 0, 0, 0],
                       [0, 0, 0, 0]])

desiredAcc = np.array([[0, 0, 0, 0],
                       [0, 0, 0, 0]])

desiredTimes = np.array([0, 10, 20, 40])
# desiredTimes = np.array([0, 20])


def waypoint_calculation(desiredPos, desiredVel, desiredAcc, desiredTimes):
    for i in range(0, 4):
        # find the shape of the desiredPos array (equivalent to size() in Matlab)
        arrayShape = np.shape(desiredPos)    
        tempKinematics = desiredPos[:,i]
        tempKinematics = np.append(tempKinematics, desiredVel[:,i], axis = 0)
        tempKinematics = np.append(tempKinematics, desiredAcc[:,i], axis = 0)
        # use loop to take out each element separately in tempKinematics
        desiredKinematics = np.zeros((np.size(tempKinematics), 1))    
        for k in range(0, np.size(tempKinematics)):
            desiredKinematics[k][0] = tempKinematics[k]

        # just the mapping coefficients mapping the initial position
        coeffMapMatrix = np.array([[1, 0, 0, 0, 0, 0]])                               
        # insert the mapping coefficients for intermediate points
        if arrayShape[0] > 2:
            for j in range(1, np.size(desiredTimes)-1):
                temp = [[1, desiredTimes[j], pow(desiredTimes[j], 2), pow(desiredTimes[j], 3), pow(desiredTimes[j], 4), pow(desiredTimes[j], 5)]]
                coeffMapMatrix = np.append(coeffMapMatrix, temp, axis = 0)
        # add the final desired position
        coeffMapMatrix = np.append(coeffMapMatrix, [[1, desiredTimes[-1], pow(desiredTimes[-1], 2), pow(desiredTimes[-1], 3), pow(desiredTimes[-1], 4), pow(desiredTimes[-1], 5)]], axis = 0)
        # add the velocity and acceleration terms
        temp2 = [[0, 1, 0, 0, 0, 0],
                [0, 1, 2*desiredTimes[-1], 3*pow(desiredTimes[-1], 2), 4*pow(desiredTimes[-1], 3), 5*pow(desiredTimes[-1], 4)],
                [0, 0, 2, 0, 0, 0],
                [0, 0, 2, 6*desiredTimes[-1], 12*pow(desiredTimes[-1], 2), 20*pow(desiredTimes[-1], 3)]]
        coeffMapMatrix = np.append(coeffMapMatrix, temp2, axis = 0)

        # perform the mapping from desired points to coefficients for sub-optimal minimum jerk waypoints
        if i == 0:
            coeffVector = np.dot(np.linalg.pinv(coeffMapMatrix), desiredKinematics)
        else:
            coeffVector = np.append(coeffVector, np.dot(np.linalg.pinv(coeffMapMatrix), desiredKinematics), axis = 1)    

    # create time vector
    timeVec = np.linspace(desiredTimes[0], desiredTimes[-1], num = 200)
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
                secondLastVel[0][i] = secondLastVel[0][i] + j*coeffVector[j][i]*np.power(desiredTimes[np.size(desiredTimes)-2], j-1)
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
                secondLastAcc[0][i] = secondLastAcc[0][i] + multFactor*coeffVector[j][i]*np.power(desiredTimes[np.size(desiredTimes)-2], j-2)
    
    # take the desired vel and accel from the 2nd to last waypoint and use as initial conditions for optimal min jerk trajectory
    # 2nd to last point index        
    minDesiredPos = np.array([desiredPos[np.size(desiredTimes)-2,:],
                              desiredPos[np.size(desiredTimes)-1,:]])
    minDesiredVel = np.concatenate((secondLastVel, np.array([[desiredVel[1,0], desiredVel[1,1], desiredVel[1,2], desiredVel[1,3]]])), axis = 0)
    minDesiredAcc = np.concatenate((secondLastAcc, np.array([[desiredAcc[1,0], desiredAcc[1,1], desiredAcc[1,2], desiredAcc[1,3]]])), axis = 0)
    timeDiff = desiredTimes[-1] - desiredTimes[np.size(desiredTimes)-2]
    
    lastPtsCoeffMapMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                      [1, timeDiff, pow(timeDiff, 2), pow(timeDiff, 3), pow(timeDiff, 4), pow(timeDiff, 5)],
                                      [0, 1, 0, 0, 0, 0],
                                      [0, 1, 2*timeDiff, 3*pow(timeDiff, 2), 4*pow(timeDiff, 3), 5*pow(timeDiff, 4)],
                                      [0, 0, 2, 0, 0, 0],
                                      [0, 0, 2, 6*timeDiff, 12*pow(timeDiff, 2), 20*pow(timeDiff, 3)]])
    for n in range(0, 4):
        desiredKinematics = np.array([minDesiredPos[:,n]]).T
        desiredKinematics = np.append(desiredKinematics, np.array([minDesiredVel[:,n]]).T, axis = 0)
        desiredKinematics = np.append(desiredKinematics, np.array([minDesiredAcc[:,n]]).T, axis = 0)
        # perform the mapping from desired points to coefficients for optimal minimum jerk waypoints on last two points
        if n == 0:
            coeffVector2 = np.dot(np.linalg.inv(lastPtsCoeffMapMatrix), desiredKinematics)
        else:
            coeffVector2 = np.append(coeffVector2, np.dot(np.linalg.inv(lastPtsCoeffMapMatrix), desiredKinematics), axis = 1) 

    indexShift = np.where(timeVec == desiredTimes[np.size(desiredTimes)-2])
    for i in range(0, 4):
        for m in range(0, np.size(timeVec)):
            # if timeVec[m] <= desiredTimes[np.size(desiredTimes)-2]:
            if timeVec[m] <= 20.12:
                coeffVectorApp = coeffVector
                indexShift = 0
            else:
                coeffVectorApp = coeffVector2
                # indexShift = np.where(timeVec == desiredTimes[np.size(desiredTimes)-2])
                indexShift = 101
            # position waypoints
            for j in range(0, 6):
                if j == 0:
                    waypoints[m,i] = coeffVectorApp[j][i]
                else:
                    waypoints[m,i] = waypoints[m,i] + coeffVectorApp[j][i]*pow(timeVec[m - indexShift], j)
            # velocity waypoints
            for j in range(1, 6):
                if j == 1:
                    desVel[m,i] = coeffVectorApp[j][i]
                else:
                    # don't need extra variable for multiplication factor when taking derivative of the position waypoints equation can just use j
                    desVel[m,i] = desVel[m,i] + j*coeffVectorApp[j][i]*pow(timeVec[m - indexShift], j-1)
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
                    desAcc[m,i] = desAcc[m,i] + multFactor*coeffVectorApp[j][i]*pow(timeVec[m - indexShift], j-2)

    return waypoints, desVel, desAcc, timeVec

# have plots been made?
plotState = False
waypoints, desVel, desAcc, timeVec = waypoint_calculation(desiredPos, desiredVel, desiredAcc, desiredTimes)
# plot the waypoints    
figPos = plt.figure()
axPos = plt.axes(projection = '3d')
axPos.plot3D(desiredPos[:,0], desiredPos[:,1], desiredPos[:,2], 'ro')

pnt3d = axPos.scatter(waypoints[:,0], waypoints[:,1], waypoints[:,2], c = timeVec)
cbar = plt.colorbar(pnt3d)
cbar.set_label("Time [sec]")
# label the axes and give title
axPos.set_xlabel('X-Axis [m]')
axPos.set_ylabel('Y-Axis [m]')
axPos.set_zlabel('Z-Axis [m]')
axPos.set_title('Sub-Optimal Minimum Jerk Position Waypoints')

# plot the desired kinematics
figOtherKinematics = plt.figure()
figOtherKinematics.suptitle('Desired Kinematics in Inertial Frame')
# desired position waypoints
axPos = plt.subplot(311)
axPos.plot(timeVec, waypoints[:,0], '-r', label = '$x_b$')
axPos.plot(timeVec, waypoints[:,1], '-k', label = '$y_b$')
axPos.plot(timeVec, waypoints[:,2], '-b', label = '$z_b$')
# add the yaw legend
axPos.plot(np.nan, '.-g', label = 'yaw')
axPos.legend(loc = 0)
plt.grid()
plt.xlabel('Time [sec]')
plt.ylabel('Position [m]')
# plt.title('Desired Position in Inertial Frame')
# desired yaw
axYaw = axPos.twinx()
axYaw.plot(timeVec, waypoints[:,3], '.-g')
axYaw.set_ylabel('Yaw [rad]')

# desired velocity waypoints
axVel = plt.subplot(312)
axVel.plot(timeVec, desVel[:,0], '-r', label = '$v_{x,b}$')
axVel.plot(timeVec, desVel[:,1], '-k', label = '$v_{y,b}$')
axVel.plot(timeVec, desVel[:,2], '-b', label = '$v_{z,b}$')
# add the yaw legend
axVel.plot(np.nan, '.-g', label = '$yaw_{rate}$')
axVel.legend(loc = 0)
plt.grid()
plt.xlabel('Time [sec]')
plt.ylabel('Velocity [m/s]')
# plt.title('Desired Velocity in Inertial Frame')
# desired yaw
axYawRate = axVel.twinx()
axYawRate.plot(timeVec, desVel[:,3], '.-g')
axYawRate.set_ylabel('Yaw [rad/s]')

# desired acceleration waypoints
axAcc = plt.subplot(313)
axAcc.plot(timeVec, desAcc[:,0], '-r', label = '$a_{x,b}$')
axAcc.plot(timeVec, desAcc[:,1], '-k', label = '$a_{y,b}$')
axAcc.plot(timeVec, desAcc[:,2], '-b', label = '$a_{z,b}$')
# add the yaw legend
axAcc.plot(np.nan, '.-g', label = '$yaw_{acc}$')
axAcc.legend(loc = 0)
plt.grid()
plt.xlabel('Time [sec]')
plt.ylabel('Acceleration [$m/s^2$]')
# plt.title('Desired Acceleration in Inertial Frame')
# desired yaw
axYawRate = axAcc.twinx()
axYawRate.plot(timeVec, desAcc[:,3], '.-g')
axYawRate.set_ylabel('Yaw [$ad/s^2$]')
plt.show()




            