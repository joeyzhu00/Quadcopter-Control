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
from waypoint_generation_library import WaypointGen

"""
TODO: Implement minimum jerk trajectory generator for waypoints
TODO: Design waypoint selector, probably just find the closest point in terms of distance
TODO: Trajectory generator with waypoints as inputs
TODO: Create function to calculate desired attitude and linear accelerations
""" 


class PDControl(object):
    """ Takes IMU and position data and publishes actuator commands based off a Proportional Derivative law"""
    def __init__(self):
        self.dlqrPublisher = rospy.Publisher("/hummingbird/command/motor_speed", Actuators, queue_size = 1)
        
        self.receivedImuQuat = Quaternion()

        self.thrustConstant = 8.54858e-06
        self.momentConstant = 1.6e-2
        self.g = 9.81    # [m/s^2]
        self.m = 0.716   # [kg]
        self.Ixx = 0.007 # [kg*m^2]
        self.Iyy = 0.007 # [kg*m^2]
        self.Izz = 0.012 # [kg*m^2]
        gamma = self.thrustConstant / self.momentConstant
        self.L = 0.17    # [m]
        # damping ratio (critically damped)
        zeta = 1
        # natural frequency
        wn = 9 # [rad/s]

        # attitude control gain calculation based on 2nd order system
        # proportional gain
        self.kpAngle = np.array(([Ixx*pow(wn,2)], # roll
                                 [Iyy*pow(wn,2)], # pitch
                                 [Izz*pow(wn,2)]) # yaw
        # derivative gain
        self.kdAngle = np.array(([Ixx*2*zeta*wn],   # roll
                                 [Iyy*2*zeta*wn],   # pitch
                                 [Izz*2*zeta*wn]])) # yaw
        
        # position desired gain hand-tuned
        # proportional gain
        self.kpPos = np.array(([1],
                               [1],
                               [1]))
        self.kdPos = np.array(([0.1],
                               [0.1],
                               [0.1]))

        # variable to keep track of the previous error in each state
        self.prevErr = np.zeros((12,1))
        # variable to store waypoints [x, y, z, yaw]
        self.waypointTarget = np.zeros((4,1))

        self.equilibriumInput = np.zeros((4,1))
        self.equilibriumInput[0] = self.m*self.g
        self.PI = 3.14159
        self.speedAllocationMatrix = np.array([[self.thrustConstant, self.thrustConstant, self.thrustConstant, self.thrustConstant],
                                               [0,                 L*self.thrustConstant,  0,                (-1)*L*self.thrustConstant],
                                               [(-1)*L*self.thrustConstant,  0,          L*self.thrustConstant, 0],
                                               [self.momentConstant, (-1)*self.momentConstant, self.momentConstant, (-1)*self.momentConstant]])
        # variable to check whether first pass has been completed to start calculating "dt"
        self.firstPass = False

        # calculate the time difference
        timeNow = rospy.get_rostime()
        # time now subtracted by start time
        self.startTime = (timeNow.secs + 1e9*timeNow.nsecs)

        # generate the waypoints
        WaypointGeneration = WaypointGen()
        self.waypoints, self.desVel, self.desAcc, self.timeVec = WaypointGeneration.waypoint_calculation()
                                                 
        
    def state_update(self, odomInput):
        """ Generate state vector from odometry input"""
        # create state vector
        state = np.zeros((12,1))        
        # position
        state[0] = odomInput.pose.pose.position.x
        state[1] = odomInput.pose.pose.position.y 
        state[2] = odomInput.pose.pose.position.z
        # velocity
        state[3] = odomInput.twist.twist.linear.x
        state[4] = odomInput.twist.twist.linear.y
        state[5] = odomInput.twist.twist.linear.z
        # angular position
        [roll, pitch, yaw] = euler_from_quaternion([odomInput.pose.pose.orientation.x,
                                                    odomInput.pose.pose.orientation.y, 
                                                    odomInput.pose.pose.orientation.z, 
                                                    odomInput.pose.pose.orientation.w])
        state[6] = roll
        state[7] = pitch
        state[8] = yaw
        # angular rate
        state[9] = odomInput.twist.twist.angular.x
        state[10] = odomInput.twist.twist.angular.y
        state[11] = odomInput.twist.twist.angular.z

        # if a nan is seen then set it to 0
        for i in range(0, len(state)):
            if np.isnan(state[i]):
                state[i] = 0
        self.ctrl_update(state)

    def calc_current_error(self):
        """ Find the desired state given the trajectory and PD gains and calculate current error"""
        # calculate the time difference
        timeNow = rospy.get_rostime()
        # time now subtracted by start time
        currTime = (timeNow.secs + 1e9*timeNow.nsecs) - self.startTime

        # find the closest index in timeVec corresponding to the current time
        nearestIdx = np.searchsorted(self.timeVec, currTime)

        trajectoryState = np.array(([]))

        # calculate current error
        currentErr = np.zeros((12,1))
        for i in range(0,12):
            currentErr[i] = desiredState[i] - state

        return desiredZAcc, currentErr


    def ctrl_update(self, state):
        """ Apply PD Control and then formulate motor speeds"""      
        # calculate the desired state at the current timestep
        desiredZAcc, currentErr = self.calc_current_error()
        
        desiredInput = np.array(([self.m*(desiredZAcc + self.g)],
                                 [self.Ixx*(self.kpAngle[0]*currentErr[6] + self.kdAngle[0]*currentErr[9])],
                                 [self.Iyy*(self.kpAngle[1]*currentErr[7] + self.kdAngle[1]*currentErr[10])],
                                 [self.Izz*(self.kpAngle[2]*currentErr[8] + self.kdAngle[2]*currentErr[11])]))
        # find the rotor speed for each rotor
        motorSpeeds = Actuators()                
        motorSpeeds.angular_velocities = np.zeros((4,1))
        motorSpeedTransitionVec = np.dot(np.linalg.inv(self.speedAllocationMatrix), desiredInput)
        motorSpeeds.angular_velocities = np.sqrt(np.abs(motorSpeedTransitionVec))

        self.prevErr = currentErr
        self.dlqrPublisher.publish(motorSpeeds)
    
    def pd_converter(self):
        """ Subscribe to the estimator """
        rospy.Subscriber("/hummingbird/ground_truth/odometry", Odometry, self.state_update, queue_size = 1)
        rospy.spin()

def main():
    rospy.init_node("pd_node", anonymous = False)
    pdOperator = PDControl()

    try:
        pdOperator.pd_converter()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()







