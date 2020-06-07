#!/usr/bin/env python
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

class AttitudeDetermination(object):
    """Takes IMU data and publishes an orientation estimate 
    TODO: Include pose estimation"""
    
    def __init__(self):
        """TODO: Include a launch file or xml file to grab state and covariance matrices from"""
        self.kfPublisher = rospy.Publisher("/localization/imu", Imu, queue_size = 1)
        self.receivedImuMsg = Imu()
        self.receivedImuQuat = Quaternion()

        # initialize initial state covariance matrix
        self.previousPosteriorState = np.identity(6)*0.01

        # initialize the previous X State
        self.previousXState = np.zeros([6,1])

        self.processVariance = 0.01
        self.measurementVariance = 0.01

        self.firstMeasurementPassed = False
        self.initTimeDelta = 0.2

    def imu_callback(self, imuMsg):
        """Callback for the imu input"""
        imuEstimate = self.kf_estimation(imuMsg)
        self.kfPublisher.publish(imuEstimate)
    
    def kf_estimation(self, imuMsg):
        """Use kf to create a state estimate"""
        # TODO: Include more states, currently only includes orientation and angular velocity states

        (imuRoll, imuPitch, imuYaw) = euler_from_quaternion([imuMsg.orientation.x, imuMsg.orientation.y, imuMsg.orientation.z, imuMsg.orientation.w])
        
        # get the current time
        currentTime = rospy.get_time()

        # calculate the time delta b/t previous measurement and current measurement
        if self.firstMeasurementPassed:
            TS = currentTime - self.previousTime
        else:
            TS = self.initTimeDelta

        # system measurements
        z = np.array([imuRoll, imuPitch, imuYaw,imuMsg.angular_velocity.x, imuMsg.angular_velocity.y, imuMsg.angular_velocity.z]).reshape(-1,1)

        # state matrix
        A = np.array([[1, 0, 0, TS, 0, 0],
                    [0, 1, 0, 0, TS, 0],
                    [0, 0, 1, 0, 0, TS],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1]])
        
        # measurement matrix
        C = np.identity(6)

        # process variance matrix
        Q = self.processVariance*np.identity(6)

        # measurement Variance
        R = self.measurementVariance*np.identity(6)

        # state matrix
        x = np.matmul(A, self.previousXState)

        # posterior state
        p = np.matmul(A, np.matmul(self.previousPosteriorState, A)) + Q

        # innovation
        S = np.linalg.inv(np.matmul(C, np.matmul(p, np.transpose(C))) + R)
        
        # kalman gain
        kalmanGain = np.matmul(p, np.matmul(np.transpose(C), S))

        # state matrix update
        xStateUpdate = x + np.matmul(kalmanGain, (z-np.matmul(C, x)))
        self.previousXState = xStateUpdate

        # posterior state update
        posteriorStateUpdate = np.matmul(np.identity(6) - np.matmul(kalmanGain, C), p)
        self.previousPosteriorState = posteriorStateUpdate

        # first pass has been completed, time delta can be updated
        self.firstMeasurementPassed = True

        # update the previous time
        self.previousTime = currentTime

        return self.imu_msg_creation(xStateUpdate)

    def imu_msg_creation(self, xState):
        """Temporary function to create an imu message given prerequisite data"""
        createdImuMsg = Imu()
        
        # orientation
        (createdImuMsg.orientation.x, createdImuMsg.orientation.y, createdImuMsg.orientation.z, createdImuMsg.orientation.w) = quaternion_from_euler(xState.item(0), xState.item(1), xState.item(2))
        
        # angular velocity
        createdImuMsg.angular_velocity.x = xState.item(3)
        createdImuMsg.angular_velocity.y = xState.item(4)
        createdImuMsg.angular_velocity.z = xState.item(5)
        
        createdImuMsg.header.frame_id = 'body'
        # add covariance data
        return createdImuMsg

    def imu_data_converter(self):
        """Subscribe to the IMU and Odometry data"""
        # TODO: Add odometry data subscriber
        # rospy.Subscriber("/hummingbird/ground_truth/imu", Imu, self.imu_callback, queue_size = 1)
        rospy.Subscriber("/imu", Imu, self.imu_callback, queue_size = 1)
        rospy.spin()

def main():
    rospy.init_node("kf_node", anonymous = False)
    kfEstimator = AttitudeDetermination()

    try:
        kfEstimator.imu_data_converter()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()
