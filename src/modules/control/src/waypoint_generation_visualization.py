#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
import rospy
import rosbag
import numpy as np
import matplotlib.pyplot as plt
from waypoint_generation_library import WaypointGen
from scipy import linalg
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from mpl_toolkits import mplot3d

WaypointGeneration = WaypointGen()
waypoints, desVel, desAcc, timeVec = WaypointGeneration.waypoint_calculation()
desiredPos = WaypointGeneration.desiredPos

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




            