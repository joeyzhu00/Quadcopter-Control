#!/usr/bin/env python3
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
from mav_msgs.msg import Actuators
from waypoint_generation_library import WaypointGen 

""" NOTE: Quaternion operations are referenced from Chapter 1 of Spacecraft Dynamics by Kane, Levinson, and Likins
          qw is not a part of the state due to lack of controllability"""

class InfDiscreteLQR(object):
    """ Takes IMU and position data and publishes actuator commands based off an infinite horizon discrete LQR control law"""
    def __init__(self):
        self.dlqrPublisher = rospy.Publisher("/hummingbird/command/motor_speed", Actuators, queue_size = 1)
        
        self.receivedImuQuat = Quaternion()

        self.thrustConstant = 8.54858e-06
        self.momentConstant = 1.6e-2
        g = 9.81    # [m/s^2]
        m = 0.716   # [kg]
        Ixx = 0.007 # [kg*m^2]
        Iyy = 0.007 # [kg*m^2]
        Izz = 0.012 # [kg*m^2]
        gamma = self.thrustConstant / self.momentConstant
        dt = 0.01   # [sec]
        L = 0.17    # [m]
        # state update matrix
        self.A = np.array([[1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 2*dt*g, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, -2*dt*g, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        # input matrix
        self.B = np.array([[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [dt/m, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, dt/Ixx, 0, 0],
                           [0, 0, dt/Iyy, 0],
                           [0, 0, 0, dt/Izz]])
        # output matrix
        self.C = np.array(([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]))

        self.equilibriumInput = np.zeros((4,1))
        self.equilibriumInput[0] = m*g
        self.PI = 3.14159
        self.speedAllocationMatrix = np.array([[self.thrustConstant, self.thrustConstant, self.thrustConstant, self.thrustConstant],
                                               [0,                 L*self.thrustConstant,  0,                (-1)*L*self.thrustConstant],
                                               [(-1)*L*self.thrustConstant,  0,          L*self.thrustConstant, 0],
                                               [self.momentConstant, (-1)*self.momentConstant, self.momentConstant, (-1)*self.momentConstant]])

        QMult = 1
        self.Q = QMult*np.eye(12)
        self.Q[2][2] = 500/QMult
        self.Q[8][8] = 10000/QMult
        self.R = 1000*np.array([[1, 0, 0, 0],
                                [0, 5, 0, 0],
                                [0, 0, 5, 0],
                                [0, 0, 0, 0.00001]])
        self.Uinf = linalg.solve_discrete_are(self.A, self.B, self.Q, self.R, None, None)
        self.trackingHorizon = 3
        # time now subtracted by start time
        self.startTime = rospy.get_time()
        # generate the waypoints
        WaypointGeneration = WaypointGen()
        self.waypoints, self.desVel, self.desAcc, self.timeVec = WaypointGeneration.waypoint_calculation()
        self.desiredPos = WaypointGeneration.desiredPos
        self.desiredTimes = WaypointGeneration.desiredTimes

        self.attitudeControlOnly = 0
        # deadbands [x-pos, y-pos, z-pos, qz]
        self.waypointDeadband = np.array(([0.3, 0.3, 0.5, 1*self.PI/180]))

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
        state[6] = odomInput.pose.pose.orientation.x
        state[7] = odomInput.pose.pose.orientation.y
        state[8] = odomInput.pose.pose.orientation.z
        # angular rate
        state[9] = odomInput.twist.twist.angular.x
        state[10] = odomInput.twist.twist.angular.y
        state[11] = odomInput.twist.twist.angular.z

        # if a nan is seen then set it to 0
        for i in range(0, len(state)):
            if np.isnan(state[i]):
                state[i] = 0
        self.ctrl_update(state, odomInput.pose.pose.orientation.w)

    def calc_error(self, state, qw):
        """ Find the desired state given the trajectory and PD gains and calculate current error"""                                 
        # calculate the time difference
        # time now subtracted by start time
        currTime = rospy.get_time() - self.startTime
        # find the closest index in timeVec corresponding to the current time
        nearestIdx = np.searchsorted(self.timeVec, currTime)
        if nearestIdx >= np.size(self.timeVec):
            nearestIdx = np.size(self.timeVec)-1        
        
        # desired quaternion (desired attitude in inertial frame)
        des_q_N = QuatMath().euler_to_quaternion(0, 0, self.waypoints[nearestIdx,3])
        # current quaternion (current attitude in inertial frame)
        b_q_N = np.array(([state[6,0]],
                          [state[7,0]],
                          [state[8,0]],
                          [qw]))
        # body attitude in desired frame
        b_q_des = QuatMath().quat_mult(b_q_N, QuatMath().quat_inverse(des_q_N))
        # current error
        currErr = np.array(([state[0,0] - self.waypoints[nearestIdx,0],
                             state[1,0] - self.waypoints[nearestIdx,1],
                             state[2,0] - self.waypoints[nearestIdx,2],
                             state[3,0] - self.desVel[nearestIdx,0],
                             state[4,0] - self.desVel[nearestIdx,1],
                             state[5,0] - self.desVel[nearestIdx,2],
                             b_q_des[2,0],
                             state[11,0] - self.desVel[nearestIdx,3]])) 
        
        # apply deadbands when reaching the final waypoint 
        if nearestIdx == (np.size(self.timeVec)-1):
            # x-pos and y-pos deadband check
            # x-pos deadband check
            if (currErr[0] <= self.waypointDeadband[0]) and (currErr[0] >= (-1)*self.waypointDeadband[0]):
                currErr[0] = 0
            # y-pos deadband check
            if (currErr[1] <= self.waypointDeadband[1]) and (currErr[1] >= (-1)*self.waypointDeadband[1]):
                currErr[1] = 0
            # z-pos deadband check
            if (currErr[2] <= self.waypointDeadband[2]) and (currErr[2] >= (-1)*self.waypointDeadband[2]):
                currErr[2] = 0
            # # yaw deadband check
            # if (currErr[6] <= self.waypointDeadband[3]) and (currErr[6] >= (-1)*self.waypointDeadband[3]):
            #     currErr[6] = 0
        # only do-pos z and yaw control
        # if self.attitudeControlOnly:
        #     print('Attitude Control Only State')
        #     currErr[0] = 0
        #     currErr[1] = 0
        return currErr

    def ctrl_update(self, state, qw):
        """ Multiply state by Discrete LQR Gain Matrix and then formulate motor speeds"""
        currErr = self.calc_error(state, qw)
        for i in range(5):
            state[i,0] = currErr[i]
        state[8,0] = currErr[6]
        state[11,0] = currErr[7]

        desiredInput = (-1)*np.dot(self.dlqrGain, state) + self.equilibriumInput
        # find the rotor speed for each rotor
        motorSpeeds = Actuators()                
        motorSpeeds.angular_velocities = np.zeros((4,1))
        motorSpeedTransitionVec = np.dot(np.linalg.inv(self.speedAllocationMatrix), desiredInput)
        motorSpeeds.angular_velocities = np.sqrt(np.abs(motorSpeedTransitionVec))

        self.dlqrPublisher.publish(motorSpeeds)
    
    def dlqr_converter(self):
        """ Subscribe to the estimator """
        # rospy.Subscriber("/hummingbird/ground_truth/odometry", Odometry, self.state_update, queue_size = 1)
        rospy.Subscriber("/localization/odom", Odometry, self.state_update, queue_size = 1)
        rospy.spin()

def main():
    rospy.init_node("inf_dlqr_node", anonymous = False)
    dlqrOperator = InfDiscreteLQR()

    try:
        dlqrOperator.dlqr_converter()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()
