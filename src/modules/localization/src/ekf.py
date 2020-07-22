#!/usr/bin/env python3
from __future__ import print_function
from __future__ import division
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion, quaternion_from_euler

"""
NOTE: Estimates the orientation and angular velocity of the quadcopter with use of the gyroscope and the orientation sensor. 

TODO: Add noise model for the gyroscope and the accelerometer, eventually expand to position estimation as well

    Subscribes
    ----------
    Topic: /hummingbird/ground_truth/imu

    Publishes
    ---------
    Topic: /localization/imu

"""

class ekfStateEstimation(object):
    """Takes IMU data and publishes an orientation estimate 
    TODO: Include pose estimation"""
    
    def __init__(self):
        """TODO: Include a launch file or xml file to grab state and covariance matrices from"""
        self.ekfPublisher = rospy.Publisher("/localization/odom", Odometry, queue_size = 1)
        self.receivedImuMsg = Imu()
        self.receivedImuQuat = Quaternion()

        # initialize initial state covariance matrix
        self.previousPm = np.identity(12)*0.01

        # initialize the previous X State
        self.previousXm = np.zeros((12,1))

        self.processVariance = 0.01
        self.measurementVariance = 0.01

        self.firstMeasurementPassed = False
        self.initTimeDelta = 0.01

        # constants
        self.g = 9.8     # [m/s^2]
        self.m = 0.716   # [kg]
        self.Ixx = 0.007 # [kg*m^2]
        self.Iyy = 0.007 # [kg*m^2]
        self.Izz = 0.012 # [kg*m^2]
        self.L = 0.17    # [m]


    def imu_callback(self, imuMsg):
        """ Callback for the imu input"""
        imuEstimate = self.imu_ekf_estimation(imuMsg)
        self.ekfPublisher.publish(imuEstimate)
    
    def pose_callback(self, poseMsg):
        """ Callback for the pose input"""

    def ctrl_input_callback(self, ctrlInput):
        """ Callback for the control input"""

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
        
        linAccel = gravityComponent + ((input[0,0] + self.m*self.g)/self.m)*rotMatThirdCol
        linVel = prevLinVel + linAccel*dt
        linPos = prevLinPos + linVel*dt
        nonLinState = np.vstack((linPos, linVel, angPos, angVel))

        return nonLinState

    def pose_ekf_estimation(self, poseMsg):
        """ Use ekf to create a state estimate when given pose and control input data"""

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
        self.controlInput = np.zeros((4,1))

        # system measurements, stuff linear acceleration into linear velocity state
        z = np.array(([self.previousXm[3,0] + linAccelInertial[0,0]*dt],
                      [self.previousXm[4,0] + linAccelInertial[1,0]*dt],
                      [self.previousXm[5,0] + (linAccelInertial[2,0] + self.g)*dt],
                      [imuRoll],
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
        # input matrix
        B = np.array([[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [dt/self.m, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, dt/self.Ixx, 0, 0],
                      [0, 0, dt/self.Iyy, 0],
                      [0, 0, 0, dt/self.Izz]])
        # measurement matrix
        H = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        # measurement noise partial derivative matrix
        M = np.identity(9)
        # process noise partial derivative matrix
        L = np.identity(12)

        # process variance matrix
        V = self.processVariance*np.identity(12)

        # measurement Variance
        W = self.measurementVariance*np.identity(9)

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

        createdOdomMsg.header.frame_id = 'body'
        
        return createdOdomMsg

    def data_converter(self):
        """ Subscribe to the IMU and Pose data"""
        # TODO: Add odometry data subscriber
        rospy.Subscriber("/hummingbird/ground_truth/imu", Imu, self.imu_callback, queue_size = 1)
        # rospy.Subscriber("/imu", Imu, self.imu_callback, queue_size = 1)
        rospy.spin()

def main():
    rospy.init_node("ekf_node", anonymous = False)
    ekfEstimator = ekfStateEstimation()

    try:
        ekfEstimator.data_converter()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()
