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
# TODO: make this critically damped by tuning the natural frequency
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
        
        # damping ratio (overdamped)
        zeta = 2
        # natural frequency
        self.PI = 3.14159
        wnAng = 35 # [rad/s]
        # attitude control gains calculation based on 2nd order system assumption
        # proportional gain
        self.kpAngle = np.array(([self.Ixx*pow(wnAng,2), # roll
                                  self.Iyy*pow(wnAng,2), # pitch
                                  self.Izz*pow(wnAng,2)])) # yaw
        print(self.kpAngle)
        # derivative gain
        self.kdAngle = np.array(([self.Ixx*2*zeta*wnAng,   # roll
                                  self.Iyy*2*zeta*wnAng,   # pitch
                                  self.Izz*2*zeta*wnAng])) # yaw
        print(self.kdAngle)

        # position control gains hand-tuned
        # proportional gain
        self.kpPos = np.array(([0.1,
                                0.1,
                                1]))
        # derivative gain
        self.kdPos = np.array(([0.1,
                                0.1,
                                1]))

        # variable to keep track of the previous error in each state
        self.prevRPYErr = np.zeros((3,1))

        
        self.speedAllocationMatrix = np.array([[self.thrustConstant, self.thrustConstant, self.thrustConstant, self.thrustConstant],
                                               [0,                 self.L*self.thrustConstant,  0,                (-1)*self.L*self.thrustConstant],
                                               [(-1)*self.L*self.thrustConstant,  0,          self.L*self.thrustConstant, 0],
                                               [self.momentConstant, (-1)*self.momentConstant, self.momentConstant, (-1)*self.momentConstant]])
        # variable to check whether first pass has been completed to start calculating "dt"
        self.firstPass = False
        # first pass dt corresponding to 100 hz controller
        self.firstPassDt = 0.01
        # time now subtracted by start time
        self.startTime = rospy.get_time()
        # previous time placeholder
        self.prevTime = 0
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

    def calc_error(self, state):
        """ Find the desired state given the trajectory and PD gains and calculate current error"""
        # calculate the time difference
        # time now subtracted by start time
        currTime = rospy.get_time() - self.startTime
        # time difference
        if not self.firstPass:
            dt = self.firstPassDt
            self.firstPass = True
        else:
            dt = currTime - self.prevTime
        # get_time() is pretty unreliable... 
        if dt <= 0.0001:
            dt = 0.01
        # find the closest index in timeVec corresponding to the current time
        nearestIdx = np.searchsorted(self.timeVec, currTime)
        if nearestIdx >= np.size(self.timeVec):
            nearestIdx = np.size(self.timeVec)-1
        print(nearestIdx)
        # desired linear acceleration calculation 
        posErr = np.array(([self.waypoints[nearestIdx,0] - state[0,0],
                            self.waypoints[nearestIdx,1] - state[1,0],
                            self.waypoints[nearestIdx,2] - state[2,0]]))

        rateErr = np.array(([self.desVel[nearestIdx,0] - state[3,0],
                             self.desVel[nearestIdx,1] - state[4,0],
                             self.desVel[nearestIdx,2] - state[5,0]]))        

        desiredLinAcc = np.array(([self.desAcc[nearestIdx,0] + self.kpPos[0]*posErr[0] + self.kdPos[0]*rateErr[0],
                                   self.desAcc[nearestIdx,1] + self.kpPos[1]*posErr[1] + self.kdPos[1]*rateErr[1],
                                   self.desAcc[nearestIdx,2] + self.kpPos[2]*posErr[2] + self.kdPos[2]*rateErr[2]]))  
        desiredZAcc = desiredLinAcc[2]
        # desired RPY angles
        rpyDes = np.array(([(1/self.g)*(desiredLinAcc[0]*np.sin(self.waypoints[nearestIdx,3]) - desiredLinAcc[1]*np.cos(self.waypoints[nearestIdx,3])),
                            (1/self.g)*(desiredLinAcc[0]*np.cos(self.waypoints[nearestIdx,3]) + desiredLinAcc[1]*np.sin(self.waypoints[nearestIdx,3])),
                            self.waypoints[nearestIdx,3]]))
        # print(rpyDes)
        # RPY error
        rpyErr = np.array(([rpyDes[0] - state[6,0],
                            rpyDes[1] - state[7,0],
                            rpyDes[2] - state[8,0]]))
        # RPY Rate error backward difference for roll and pitch
        rpyRateErr = np.array(([(rpyErr[0] - self.prevRPYErr[0])/dt,
                                (rpyErr[1] - self.prevRPYErr[1])/dt,
                                self.desVel[nearestIdx, 3] - state[11,0]]))
        # record the time
        self.prevTime = currTime
        # record the error
        self.prevRPYErr = rpyErr
        return desiredZAcc, rpyErr, rpyRateErr

    def ctrl_update(self, state):
        """ Apply PD Control and then formulate motor speeds"""      
        # calculate the desired state at the current timestep
        desiredZAcc, rpyErr, rpyRateErr = self.calc_error(state)
        
        # print(rpyErr[0])
        desiredInput = np.array(([self.m*(desiredZAcc + self.g)],
                                 [self.Ixx*(self.kpAngle[0]*rpyErr[0] + self.kdAngle[0]*rpyRateErr[0])],
                                 [self.Iyy*(self.kpAngle[1]*rpyErr[1] + self.kdAngle[1]*rpyRateErr[1])],
                                 [self.Izz*(self.kpAngle[2]*rpyErr[2] + self.kdAngle[2]*rpyRateErr[2])]))

        print(desiredInput)
        # find the rotor speed for each rotor
        motorSpeeds = Actuators()                
        motorSpeeds.angular_velocities = np.zeros((4,1))
        motorSpeedTransitionVec = np.dot(np.linalg.inv(self.speedAllocationMatrix), desiredInput)
        motorSpeeds.angular_velocities = np.sqrt(np.abs(motorSpeedTransitionVec))
    
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







