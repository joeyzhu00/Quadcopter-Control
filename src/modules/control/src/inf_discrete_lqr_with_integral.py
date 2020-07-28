#!/usr/bin/env python3
from __future__ import print_function
from __future__ import division
import rospy
import rosbag
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.linalg import block_diag
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from mav_msgs.msg import Actuators
from waypoint_generation_library import WaypointGen 

class InfDiscreteLQRWithIntegrator(object):
    """ Takes IMU and position data and publishes actuator commands based off an infinite horizon discrete LQR control law with integrator"""
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
        # output matrix
        C = np.array(([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
        # enlarged state matrix with the error accumulation as a state
        Ag = np.vstack((np.hstack((A, np.zeros((12,8)))), np.hstack((C, np.eye(8)))))
        # enlarged input matrix with the error accumulation as a state
        Bg = np.vstack((B, np.zeros((8,4))))

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
        # state cost with enlarged state vector
        Qint = np.eye(8)
        Qg = block_diag(Q, Qint)
        R = 1000*np.array([[1, 0, 0, 0],
                          [0, 5, 0, 0],
                          [0, 0, 5, 0],
                          [0, 0, 0, 0.00001]])
        # R = np.array([[1, 0, 0, 0],
        #               [0, 1, 0, 0],
        #               [0, 0, 1, 0],
        #               [0, 0, 0, 0.1]])
        Uinf = linalg.solve_discrete_are(Ag, Bg, Qg, R, None, None)
        # gain matrix containing both the lqr gain and the integral gain
        self.combinedGainMatrix = np.dot(np.linalg.inv(R + np.dot(Bg.T, np.dot(Uinf, Bg))), np.dot(Bg.T, np.dot(Uinf, Ag)))   
        self.dlqrGain = self.combinedGainMatrix[:,0:12]
        print('LQR Gain: ')
        print(self.dlqrGain)
        print('\n')
        self.integralGain = self.combinedGainMatrix[:,12:20]
        print('Integral Gain: ')
        print(self.integralGain)
        print('\n')
        self.previousError = np.zeros((8,1))

        # Uinf = linalg.solve_discrete_are(A, B, Q, R, None, None)
        # self.dlqrGain = np.dot(np.linalg.inv(R + np.dot(B.T, np.dot(Uinf, B))), np.dot(B.T, np.dot(Uinf, A)))   

        # time now subtracted by start time
        self.startTime = rospy.get_time()
        # generate the waypoints
        WaypointGeneration = WaypointGen()
        self.waypoints, self.desVel, self.desAcc, self.timeVec = WaypointGeneration.waypoint_calculation()
        self.desiredPos = WaypointGeneration.desiredPos
        self.desiredTimes = WaypointGeneration.desiredTimes
        
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

    def calc_error(self, state):
        """ Find the desired state given the trajectory and PD gains and calculate current error"""                                 
        integralErrFlag = 0
        # calculate the time difference
        # time now subtracted by start time
        currTime = rospy.get_time() - self.startTime
        # find the closest index in timeVec corresponding to the current time
        nearestIdx = np.searchsorted(self.timeVec, currTime)
        if nearestIdx >= np.size(self.timeVec):
            nearestIdx = np.size(self.timeVec)-1 
            integralErrFlag = 1       
        # current error
        currErr = np.array(([state[0,0] - self.waypoints[nearestIdx,0],
                             state[1,0] - self.waypoints[nearestIdx,1],
                             state[2,0] - self.waypoints[nearestIdx,2],
                             state[3,0] - self.desVel[nearestIdx,0],
                             state[4,0] - self.desVel[nearestIdx,1],
                             state[5,0] - self.desVel[nearestIdx,2],
                             state[8,0] - self.waypoints[nearestIdx,3],
                             state[11,0] - self.desVel[nearestIdx,3]]))
        errorAccum = np.reshape(currErr,(8,1)) + self.previousError
        self.previousError = np.reshape(currErr,(8,1))

        return currErr, errorAccum, integralErrFlag

    def ctrl_update(self, state):
        """ Multiply state by Discrete LQR Gain Matrix and then formulate motor speeds"""
        currErr, errorAccum, integralErrFlag = self.calc_error(state)
        for i in range(5):
            state[i,0] = currErr[i]
        state[8,0] = currErr[6]
        state[11,0] = currErr[7]
        
        if integralErrFlag:
            desiredInput = (-1)*np.dot(self.dlqrGain, state) + self.equilibriumInput + np.dot(self.integralGain, errorAccum)
        else:
            desiredInput = (-1)*np.dot(self.dlqrGain, state) + self.equilibriumInput

        # find the rotor speed for each rotor
        motorSpeeds = Actuators()                
        motorSpeeds.angular_velocities = np.zeros((4,1))
        motorSpeedTransitionVec = np.dot(np.linalg.inv(self.speedAllocationMatrix), desiredInput)
        motorSpeeds.angular_velocities = np.sqrt(np.abs(motorSpeedTransitionVec))

        self.dlqrPublisher.publish(motorSpeeds)
    
    def dlqr_converter(self):
        """ Subscribe to the estimator """
        rospy.Subscriber("/hummingbird/ground_truth/odometry", Odometry, self.state_update, queue_size = 1)
        rospy.spin()

def main():
    rospy.init_node("dlqr_node", anonymous = False)
    dlqrOperator = InfDiscreteLQRWithIntegrator()

    try:
        dlqrOperator.dlqr_converter()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()







