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
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from mav_msgs.msg import Actuators
from waypoint_generation_library import WaypointGen 

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
        A = np.array([[1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, dt*g, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, -dt*g, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        # input matrix
        B = np.array([[0, 0, 0, 0],
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

        self.equilibriumInput = np.zeros((4,1))
        self.equilibriumInput[0] = m*g
        self.PI = 3.14159
        self.speedAllocationMatrix = np.array([[self.thrustConstant, self.thrustConstant, self.thrustConstant, self.thrustConstant],
                                               [0,                 L*self.thrustConstant,  0,                (-1)*L*self.thrustConstant],
                                               [(-1)*L*self.thrustConstant,  0,          L*self.thrustConstant, 0],
                                               [self.momentConstant, (-1)*self.momentConstant, self.momentConstant, (-1)*self.momentConstant]])

        QMult = 1
        Q = QMult*np.eye(12)
        Q[2][2] = 500/QMult
        Q[8][8] = 10000/QMult
        # Q[2][2] = 100/QMult
        # Q[8][8] = 5000/QMult

        # R = 1000*np.array([[1, 0, 0, 0],
        #                   [0, 5, 0, 0],
        #                   [0, 0, 5, 0],
        #                   [0, 0, 0, 0.00001]])
        R = 1000*np.array([[1, 0, 0, 0],
                          [0, 5, 0, 0],
                          [0, 0, 5, 0],
                          [0, 0, 0, 0.1]])
        Uinf = linalg.solve_discrete_are(A, B, Q, R, None, None)
        self.dlqrGain = np.dot(np.linalg.inv(R + np.dot(B.T, np.dot(Uinf, B))), np.dot(B.T, np.dot(Uinf, A)))   

        # time now subtracted by start time
        self.startTime = rospy.get_time()
        # generate the waypoints
        WaypointGeneration = WaypointGen()
        self.waypoints, self.desVel, self.desAcc, self.timeVec = WaypointGeneration.waypoint_calculation()
        self.desiredPos = WaypointGeneration.desiredPos
        self.desiredTimes = WaypointGeneration.desiredTimes

        # deadbands [x-pos, y-pos, z-pos, yaw]
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

    def calc_pos_error(self, state):
        """ Find the desired state given the trajectory and PD gains and calculate current error"""                                 
        # calculate the time difference
        # time now subtracted by start time
        currTime = rospy.get_time() - self.startTime
        # find the closest index in timeVec corresponding to the current time
        nearestIdx = np.searchsorted(self.timeVec, currTime)
        if nearestIdx >= np.size(self.timeVec):
            nearestIdx = np.size(self.timeVec)-1        
        # current error
        currErr = np.array(([state[0,0] - self.waypoints[nearestIdx,0],
                             state[1,0] - self.waypoints[nearestIdx,1],
                             state[2,0] - self.waypoints[nearestIdx,2],
                             state[3,0] - self.desVel[nearestIdx,0],
                             state[4,0] - self.desVel[nearestIdx,1],
                             state[5,0] - self.desVel[nearestIdx,2],
                             state[8,0] - self.waypoints[nearestIdx,3],
                             state[11,0] - self.desVel[nearestIdx,3]])) 
        
        # apply deadbands when reaching the final waypoint 
        if nearestIdx >= np.size(self.timeVec):
            # x-pos, y-pos, z-pos deadband check
            for i in range(0,np.size(self.waypointDeadband)-1):
                if currErr[i] <= self.waypointDeadband[i]:
                    currErr[i] = 0
            # yaw deadband check
            if currErr[6] <= self.waypointDeadband[i]:
                currErr[i] = 0

        return currErr

    def ctrl_update(self, state):
        """ Multiply state by Discrete LQR Gain Matrix and then formulate motor speeds"""
        currErr = self.calc_pos_error(state)
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







