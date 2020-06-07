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
from mav_msgs.msg import Actuators

class LivePlot(object):
    def __init__(self):
        self.PI = 3.14159
        self.desiredState = np.zeros((12,1))
        self.desiredState[0] = 3
        self.desiredState[1] = 6
        self.desiredState[2] = 15
        self.desiredState[8] = self.PI/2
        
    
    def plot_data(self, odomMsg):
        """Function to plot reference state and current state data"""
        time = odomMsg.header.stamp.secs + odomMsg.header.stamp.nsecs*1e-9
        plt.subplot(221)
        # x-axis position            
        plt.plot(time, odomMsg.pose.pose.position.x, 'ro')
        plt.plot(time, self.desiredState[0], 'r*')
        # y-axis position            
        plt.plot(time, odomMsg.pose.pose.position.y, 'bo')
        plt.plot(time, self.desiredState[1], 'b*')
        # z-axis position            
        plt.plot(time, odomMsg.pose.pose.position.z, 'go')
        plt.plot(time, self.desiredState[2], 'g*')
        plt.xlabel('Time [sec]')
        plt.ylabel('Body Position in Inertial Frame [m]')
        plt.legend(['$x_b$','$x_{des}$', '$y_b$', '$y_{des}$', '$z_b$', '$z_{des}$'])
        plt.grid()
        # body linear velocities
        # plt.subplot(222)
        # # x-axis velocity            
        # plt.plot(time, odomMsg.twist.twist.linear.x, 'ro')
        # plt.plot(time, self.desiredState[3], 'r*')
        # # y-axis velocity            
        # plt.plot(time, odomMsg.twist.twist.linear.y, 'bo')
        # plt.plot(time, self.desiredState[4], 'b*')
        # # z-axis velocity            
        # plt.plot(time, odomMsg.twist.twist.linear.z, 'go')
        # plt.plot(time, self.desiredState[5], 'g*')
        # plt.xlabel('Time [sec]')
        # plt.ylabel('Body Velocity in Inertial Frame [m]')
        # plt.legend(['$v_b$','vdes', '$v_b$', 'vdes', '$v_b$', 'vdes'])
        # plt.grid()
        # position errors
        plt.subplot(222)
        plt.plot(time, odomMsg.pose.pose.position.x - self.desiredState[0], 'ro')
        # y-axis position            
        plt.plot(time, odomMsg.pose.pose.position.y - self.desiredState[1], 'bo')
        # z-axis position        
        plt.plot(time, odomMsg.pose.pose.position.z - self.desiredState[2], 'go')            
        plt.xlabel('Time [sec]')
        plt.ylabel('Body Position Error in Inertial Frame [m]')
        plt.legend(['$x_{err}$','$y_{err}$','$z_{err}$'])
        plt.grid()

        # body attitude
        [roll, pitch, yaw] = euler_from_quaternion([odomMsg.pose.pose.orientation.x,
                                                    odomMsg.pose.pose.orientation.y, 
                                                    odomMsg.pose.pose.orientation.z, 
                                                    odomMsg.pose.pose.orientation.w])

        plt.subplot(223)
        # roll
        plt.plot(time, roll, 'ro')
        plt.plot(time, self.desiredState[6], 'r*')
        # pitch
        plt.plot(time, pitch, 'bo')
        plt.plot(time, self.desiredState[7], 'b*')
        # yaw
        plt.plot(time, yaw, 'go')
        plt.plot(time, self.desiredState[8], 'g*')
        plt.xlabel('Time [sec]')
        plt.ylabel('Body Attitude in Inertial Frame [m]')
        plt.legend(['roll','$roll_{des}$', 'pitch', '$pitch_{des}$', 'yaw', '$yaw_{des}$'])
        plt.grid()

        # attitude error
        plt.subplot(224)
        # roll
        plt.plot(time, roll - self.desiredState[6], 'ro')
        # pitch
        plt.plot(time, pitch - self.desiredState[7], 'bo')
        # yaw
        plt.plot(time, yaw - self.desiredState[8], 'go')            
        plt.xlabel('Time [sec]')
        plt.ylabel('Body Attitude Error in Inertial Frame [m]')
        plt.legend(['$roll_{err}$','$pitch_{err}$','$yaw_{err}$'])
        plt.grid()

        plt.plot()
        plt.draw()
        plt.pause(0.0000001)

    def live_plot_operation(self):
        """ Subscribe to the estimator """
        rospy.Subscriber("/hummingbird/ground_truth/odometry", Odometry, self.plot_data, queue_size = 1)
        plt.ion()
        plt.show()
        rospy.spin()

def main():
    rospy.init_node("live_plot_node", anonymous = False)
    livePlotter = LivePlot()

    try:
        livePlotter.live_plot_operation()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()
