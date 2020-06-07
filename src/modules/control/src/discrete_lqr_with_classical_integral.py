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

"""TODO: Add integrator action (not included in LQ gain calculation but rather from classical control) but need slew planner first"""
"""TODO: Slew planner subscriber after it is designed"""
class DiscreteLQRFakeIntegrator(object):
    """ Takes IMU and position data and publishes actuator commands based off an infinite horizon discrete LQR control law in addition to integral action
        based off a simple k_i*sum(err) scheme"""
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
        self.integralGain = 1
        self.errAccumulation = 0
        Q = np.eye(12)
        Q[2][2] = 500
        Q[8][8] = 10000
        R = 100*np.array([[10, 0, 0, 0],
                      [0, 5, 0, 0],
                      [0, 0, 5, 0],
                      [0, 0, 0, 0.0001]])

        Uinf = linalg.solve_discrete_are(A, B, Q, R, None, None)
        self.dlqrGain = np.dot(np.linalg.inv(R + np.dot(B.T, np.dot(Uinf, B))), np.dot(B.T, np.dot(Uinf, A)))                                                  
        
    def state_update(self, odomInput):
        """ Generate state vector from odometry input"""
        # create state vector
        state = np.zeros((12,1))        
        # position
        state[0] = odomInput.pose.pose.position.x - 3
        state[1] = odomInput.pose.pose.position.y - 6 
        state[2] = odomInput.pose.pose.position.z - 15
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
        state[8] = yaw - self.PI/2
        # angular rate
        state[9] = odomInput.twist.twist.angular.x
        state[10] = odomInput.twist.twist.angular.y
        state[11] = odomInput.twist.twist.angular.z

        # if a nan is seen then set it to 0
        for i in range(0, len(state)):
            if np.isnan(state[i]):
                state[i] = 0
        self.ctrl_update(state)

    def ctrl_update(self, state):
        """ Multiply state by Discrete LQR Gain Matrix and then formulate motor speeds"""  
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
    dlqrOperator = DiscreteLQR()

    try:
        dlqrOperator.dlqr_converter()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()







