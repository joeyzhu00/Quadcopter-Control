#!/usr/bin/env python3
from __future__ import print_function
from __future__ import division
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Pose
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from mav_msgs.msg import Actuators

"""
TODO: Add noise model for the position sensing

    Subscribed to
    ----------
    Topic: /hummingbird/imu
           /hummingbird/ground_truth/pose (will be replaced with gps reading in real quad)
           /hummingbird/motor_speed 

    Publishes
    ---------
    Topic: /localization/odom

"""

class SimulationEkfStateEstimation(object):
    """ Takes IMU, position data, and motor speed data and generates a state estimate based off an Extended Kalman Filter"""    
    def __init__(self):
        self.ekfPublisher = rospy.Publisher("/localization/odom", Odometry, queue_size = 1)
        # initialize initial state covariance matrix
        self.previousPm = np.identity(12)*0.1

        # initialize the previous X State
        self.previousXm = np.zeros((12,1))

        self.processVariance = 0.01
        self.measurementVariance = 0.05

        self.firstMeasurementPassed = False
        self.initTimeDelta = 0.01

        # constants
        self.g = 9.8     # [m/s^2]
        self.m = 0.716   # [kg]
        self.Ixx = 0.007 # [kg*m^2]
        self.Iyy = 0.007 # [kg*m^2]
        self.Izz = 0.012 # [kg*m^2]
        self.L = 0.17    # [m]
        self.thrustConstant = 8.54858e-06
        self.momentConstant = 1.6e-2

        # matrix mapping squared motor angular velocity to force/torque control input
        self.speedAllocationMatrix = np.array([[self.thrustConstant, self.thrustConstant, self.thrustConstant, self.thrustConstant],
                                               [0,                 self.L*self.thrustConstant,  0,                (-1)*self.L*self.thrustConstant],
                                               [(-1)*self.L*self.thrustConstant,  0,          self.L*self.thrustConstant, 0],
                                               [self.momentConstant, (-1)*self.momentConstant, self.momentConstant, (-1)*self.momentConstant]])
        self.controlInput = np.zeros((4,1))

    def imu_callback(self, imuMsg):
        """ Callback for the imu input"""
        imuEstimate = self.imu_ekf_estimation(imuMsg)
        self.ekfPublisher.publish(imuEstimate)
    
    def pose_callback(self, poseMsg):
        """ Callback for the pose input"""
        poseEstimate = self.pose_ekf_estimation(poseMsg)
        self.ekfPublisher.publish(poseEstimate)

    def motor_speed_callback(self, motorSpeeds):
        """ Callback for the motor speeds"""
        motorSpeedSquaredArray = np.array(([pow(motorSpeeds.angular_velocities[0], 2)],
                                           [pow(motorSpeeds.angular_velocities[1], 2)],
                                           [pow(motorSpeeds.angular_velocities[2], 2)],
                                           [pow(motorSpeeds.angular_velocities[3], 2)]))
        # map the squared motor speeds to forces and torques
        self.controlInput = np.dot(self.speedAllocationMatrix, motorSpeedSquaredArray)
        

    def quad_nonlinear_eom(self, state, input, dt):
        """ Function for nonlinear equations of motion of quadcopter """
        # RPY position and rate update
        prevAngPos = np.array(([state[6], 
                                state[7], 
                                state[8]]))
        prevAngVel = np.array(([state[9],
                                state[10], 
                                state[11]]))  
        
        angAccel = np.array(([(input[1,0] + self.Iyy*prevAngVel[1,0]*prevAngVel[2,0] - self.Izz*prevAngVel[1,0]*prevAngVel[2,0])/self.Ixx],
                             [(input[2,0] - self.Ixx*prevAngVel[0,0]*prevAngVel[2,0] + self.Izz*prevAngVel[0,0]*prevAngVel[2,0])/self.Iyy],
                             [(input[3,0] + self.Ixx*prevAngVel[0,0]*prevAngVel[1,0] - self.Iyy*prevAngVel[0,0]*prevAngVel[1,0])/self.Izz]))
        angVel = prevAngVel + angAccel*dt
        angPos = prevAngPos + angVel*dt
        # XYZ position and rate update
        prevLinPos = np.array(([state[0], 
                                state[1], 
                                state[2]]))
        prevLinVel = np.array(([state[3],
                                state[4], 
                                state[5]]))  

        gravityComponent = np.array(([[0],
                                    [0],
                                    [-self.g]]))
        
        # 1-2-3 rotation matrix of inertial in body frame
        rotMatThirdCol = np.array(([np.cos(prevAngPos[2,0])*np.sin(prevAngPos[1,0])*np.cos(prevAngPos[0,0]) + np.sin(prevAngPos[2,0])*np.sin(prevAngPos[0,0])],
                                   [np.sin(prevAngPos[2,0])*np.sin(prevAngPos[1,0])*np.cos(prevAngPos[0,0]) - np.cos(prevAngPos[2,0])*np.sin(prevAngPos[0,0])],
                                   [np.cos(prevAngPos[1,0])*np.cos(prevAngPos[0,0])]))
        
        linAccel = gravityComponent + ((input[0,0])/self.m)*rotMatThirdCol
        linVel = prevLinVel + linAccel*dt
        linPos = prevLinPos + linVel*dt
        nonLinState = np.vstack((linPos, linVel, angPos, angVel))

        return nonLinState

    def pose_ekf_estimation(self, poseMsg):
        """ Use ekf to create a state estimate when given pose and control input data"""
        # get the current time
        currentTime = rospy.get_time()

        # calculate the time delta b/t previous measurement and current measurement
        if self.firstMeasurementPassed:
            dt = currentTime - self.previousTime
        else:
            dt = self.initTimeDelta
        if dt == 0.0:
            dt = 0.01

        # system measurements, stuff linear acceleration into linear velocity state
        z = np.array(([poseMsg.position.x],
                      [poseMsg.position.y],
                      [poseMsg.position.z]))

        # prior state 
        xp = self.quad_nonlinear_eom(self.previousXm, self.controlInput, dt)

        # state update matrix
        A = np.array([[1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, dt*self.g, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, -dt*self.g, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        # measurement matrix
        H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        # measurement noise partial derivative matrix
        M = np.identity(3)
        # process noise partial derivative matrix
        L = np.identity(12)

        # process variance matrix
        V = self.processVariance*np.identity(12)

        # measurement Variance
        W = self.measurementVariance*np.identity(3)

        # prior covariance
        Pp = np.dot(A, np.dot(self.previousPm, np.transpose(A))) + np.dot(L, np.dot(V, np.transpose(L)))

        # kalman gain
        K = np.dot(Pp, np.dot(np.transpose(H), np.linalg.inv(np.dot(H, np.dot(Pp, np.transpose(H))) + np.dot(M, np.dot(W, np.transpose(M))))))

        # state matrix update
        xm = xp + np.dot(K, (z-np.dot(H, xp)))
        self.previousXm = xm

        # posterior state update
        Pm = np.dot(np.identity(12) - np.dot(K, H), Pp)
        self.previousPm = Pm

        # first pass has been completed, time delta can be updated
        self.firstMeasurementPassed = True

        # update the previous time
        self.previousTime = currentTime

        return self.odom_msg_creation(xm)

    def imu_ekf_estimation(self, imuMsg):
        """ Use ekf to create a state estimate when given imu and control input data"""
        # convert quat to rpy angles
        (imuRoll, imuPitch, imuYaw) = euler_from_quaternion([imuMsg.orientation.x, imuMsg.orientation.y, imuMsg.orientation.z, imuMsg.orientation.w])
        
        # get the current time
        currentTime = rospy.get_time()

        # calculate the time delta b/t previous measurement and current measurement
        if self.firstMeasurementPassed:
            dt = currentTime - self.previousTime
        else:
            dt = self.initTimeDelta
        if dt == 0.0:
            dt = 0.01
        # rotate the acceleration measurements to inertial frame 
        prevRoll = self.previousXm[6,0]
        prevPitch = self.previousXm[7,0]
        prevYaw = self.previousXm[8,0]

        # body in inertial frame 1-2-3 rotation
        N_R_b = np.array(([np.cos(prevYaw)*np.cos(prevPitch), np.cos(prevYaw)*np.sin(prevPitch)*np.sin(prevRoll) - np.sin(prevYaw)*np.cos(prevRoll), np.cos(prevYaw)*np.sin(prevPitch)*np.cos(prevRoll) + np.sin(prevRoll)*np.sin(prevYaw)],
                          [np.sin(prevYaw)*np.cos(prevPitch), np.sin(prevYaw)*np.sin(prevPitch)*np.sin(prevRoll) + np.cos(prevRoll)*np.cos(prevYaw), np.sin(prevYaw)*np.sin(prevPitch)*np.cos(prevRoll) - np.cos(prevYaw)*np.sin(prevRoll)],
                          [(-1)*np.sin(prevPitch), np.cos(prevPitch)*np.sin(prevRoll), np.cos(prevPitch)*np.cos(prevRoll)]))

        linAccelBody = np.array(([imuMsg.linear_acceleration.x],
                                 [imuMsg.linear_acceleration.y],
                                 [(-1)*imuMsg.linear_acceleration.z]))
        # rotate linear acceleration in body frame into linear acceleration in inertial frame
        linAccelInertial = np.dot(N_R_b, linAccelBody)

        if self.controlInput[0,0] <= 0.05:
            linAccelInertial[2,0] = 0

        # system measurements, stuff linear acceleration into linear velocity state
        # z = np.array(([self.previousXm[3,0] + linAccelInertial[0,0]*dt],
        #               [self.previousXm[4,0] + linAccelInertial[1,0]*dt],
        #               [self.previousXm[5,0] + linAccelInertial[2,0]*dt],
        #               [imuRoll],
        #               [imuPitch],
        #               [imuYaw],
        #               [imuMsg.angular_velocity.x],
        #               [imuMsg.angular_velocity.y],
        #               [imuMsg.angular_velocity.z]))
        z = np.array(([imuRoll],
                      [imuPitch],
                      [imuYaw],
                      [imuMsg.angular_velocity.x],
                      [imuMsg.angular_velocity.y],
                      [imuMsg.angular_velocity.z]))
        # prior state 
        xp = self.quad_nonlinear_eom(self.previousXm, self.controlInput, dt)

        # state update matrix
        A = np.array([[1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, dt*self.g, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, -dt*self.g, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        # # measurement matrix
        # H = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        #               [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        #               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        #               [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        #               [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        #               [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        #               [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        #               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        #               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        # # measurement noise partial derivative matrix
        # M = np.identity(9)

        # measurement matrix
        H = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        # measurement noise partial derivative matrix
        M = np.identity(6)
        # process noise partial derivative matrix
        L = np.identity(12)

        # process variance matrix
        V = self.processVariance*np.identity(12)

        # measurement Variance
        # W = self.measurementVariance*np.identity(9)
        W = self.measurementVariance*np.identity(6)
        # prior covariance
        Pp = np.dot(A, np.dot(self.previousPm, np.transpose(A))) + np.dot(L, np.dot(V, np.transpose(L)))

        # kalman gain
        K = np.dot(Pp, np.dot(np.transpose(H), np.linalg.inv(np.dot(H, np.dot(Pp, np.transpose(H))) + np.dot(M, np.dot(W, np.transpose(M))))))

        # state matrix update
        xm = xp + np.dot(K, (z-np.dot(H, xp)))
        self.previousXm = xm

        # posterior state update
        Pm = np.dot(np.identity(12) - np.dot(K, H), Pp)
        self.previousPm = Pm

        # first pass has been completed, time delta can be updated
        self.firstMeasurementPassed = True

        # update the previous time
        self.previousTime = currentTime

        return self.odom_msg_creation(xm)

    def odom_msg_creation(self, xm):
        """ Temporary function to create an odom message given state data"""
        createdOdomMsg = Odometry()
        
        # orientation
        (createdOdomMsg.pose.pose.orientation.x, createdOdomMsg.pose.pose.orientation.y, createdOdomMsg.pose.pose.orientation.z, createdOdomMsg.pose.pose.orientation.w) = quaternion_from_euler(xm[6], xm[7], xm[8])
        
        # angular velocity
        createdOdomMsg.twist.twist.angular.x = xm[9]
        createdOdomMsg.twist.twist.angular.y = xm[10]
        createdOdomMsg.twist.twist.angular.z = xm[11]
        
        # position
        createdOdomMsg.pose.pose.position.x = xm[0]
        createdOdomMsg.pose.pose.position.y = xm[1]
        createdOdomMsg.pose.pose.position.z = xm[2]

        # linear velocity
        createdOdomMsg.twist.twist.linear.x = xm[3]
        createdOdomMsg.twist.twist.linear.y = xm[4]
        createdOdomMsg.twist.twist.linear.z = xm[5]
        
        return createdOdomMsg

    def data_converter(self):
        """ Subscribe to the IMU, pose, and motor speed data"""
        rospy.Subscriber("/hummingbird/imu", Imu, self.imu_callback, queue_size = 1)
        rospy.Subscriber("/hummingbird/ground_truth/pose", Pose, self.pose_callback, queue_size = 1)
        rospy.Subscriber("/hummingbird/motor_speed", Actuators, self.motor_speed_callback, queue_size = 1)
        rospy.spin()

def main():
    rospy.init_node("ekf_node", anonymous = False)
    ekfEstimator = SimulationEkfStateEstimation()

    try:
        ekfEstimator.data_converter()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()
