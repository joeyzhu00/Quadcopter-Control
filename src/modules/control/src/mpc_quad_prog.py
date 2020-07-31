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
import cvxpy as cv

class MPCQuadProg(object):
    """ Takes IMU and position data and publishes actuator commands based off a MPC Quadratic Program control law"""
    def __init__(self):
        self.mpcPublisher = rospy.Publisher("/hummingbird/command/motor_speed", Actuators, queue_size = 1)
        
        self.receivedImuQuat = Quaternion()

        self.thrustConstant = 8.54858e-06
        self.momentConstant = 1.6e-2
        self.g = 9.81    # [m/s^2]
        self.m = 0.716   # [kg]
        Ixx = 0.007 # [kg*m^2]
        Iyy = 0.007 # [kg*m^2]
        Izz = 0.012 # [kg*m^2]
        gamma = self.thrustConstant / self.momentConstant
        dt = 0.1   # [sec]
        L = 0.17    # [m]
        # state update matrix
        self.A = np.array([[1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0],
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
        self.B = np.array([[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [dt/self.m, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, dt/Ixx, 0, 0],
                      [0, 0, dt/Iyy, 0],
                      [0, 0, 0, dt/Izz]])
        # gravity component of input
        self.Bg = np.array([0,
                        0,
                        0,
                        0,
                        0,
                        -self.g*dt,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0])

        self.equilibriumInput = np.zeros((4))
        self.equilibriumInput[0] = self.m*self.g
        self.PI = 3.14159
        self.speedAllocationMatrix = np.array([[self.thrustConstant, self.thrustConstant, self.thrustConstant, self.thrustConstant],
                                               [0,                 L*self.thrustConstant,  0,                (-1)*L*self.thrustConstant],
                                               [(-1)*L*self.thrustConstant,  0,          L*self.thrustConstant, 0],
                                               [self.momentConstant, (-1)*self.momentConstant, self.momentConstant, (-1)*self.momentConstant]])

        QMult = 1
        self.Q = QMult*np.eye(12)
        self.Q[2][2] = 500/QMult
        self.R = 1000*np.array([[1, 0, 0, 0],
                          [0, 5, 0, 0],
                          [0, 0, 5, 0],
                          [0, 0, 0, 1]])

        self.Uinf = linalg.solve_discrete_are(self.A, self.B, self.Q, self.R, None, None)
        self.dlqrGain = np.dot(np.linalg.inv(self.R + np.dot(self.B.T, np.dot(self.Uinf, self.B))), np.dot(self.B.T, np.dot(self.Uinf, self.A)))   

        # time now subtracted by start time
        self.startTime = rospy.get_time()
        # generate the waypoints
        WaypointGeneration = WaypointGen()
        self.waypoints, self.desVel, self.desAcc, self.timeVec = WaypointGeneration.waypoint_calculation()
        self.desiredPos = WaypointGeneration.desiredPos
        self.desiredTimes = WaypointGeneration.desiredTimes

        self.mpcHorizon = 2
        # number of inputs
        self.nu = 4
        # number of states
        self.nx = 12
        self.u = cv.Variable((self.nu, self.mpcHorizon))
        self.x = cv.Variable((self.nx, self.mpcHorizon+1))

        # set up the MPC constraints
        D2R = self.PI/180
        minVal, maxVal = self.find_min_max_waypoints()
        posTolerance = 0.5 # [m]
        angTolerance = 5*D2R # [rad]
        self.xmin = np.array(([minVal[0]-posTolerance, minVal[1]-posTolerance, minVal[2]-posTolerance, -np.Inf, -np.Inf, -np.Inf, 
                        -np.Inf, -np.Inf, minVal[3]-angTolerance, -np.Inf, -np.Inf, -np.Inf]))
        self.xmax = np.array(([maxVal[0]+posTolerance, maxVal[1]+posTolerance, maxVal[2]+posTolerance, np.Inf, np.Inf, np.Inf,
                        np.Inf, np.Inf, maxVal[3]+angTolerance, np.Inf, np.Inf, np.Inf]))  
        self.umin = np.array(([-self.m*self.g, -0.5, -0.5, -0.5]))
        self.umax = np.array(([1.5*self.m*self.g, 0.5, 0.5, 0.5])) 

    def find_min_max_waypoints(self):
        """ Function to find the minimum and maximum waypoint values"""
        # find the shape of the desiredPos array (equivalent to size() in Matlab)
        arrayShape = np.shape(self.waypoints)  
        # print(arrayShape[1])
        minVal = np.zeros((arrayShape[1]))
        maxVal = np.zeros((arrayShape[1]))
        for i in range(arrayShape[1]):
            minVal[i] = min(self.waypoints[:,i])
            maxVal[i] = max(self.waypoints[:,i])

        return minVal, maxVal 
    def mpc_problem_def(self, xInit, xr):
        """ Function to setup the MPC problem given the reference state, initial state,
            corresponding infinite horizon discrete lqr gain matrices, and constraints"""
        objective = 0
        constraints = [self.x[:,0] == xInit]
        for k in range(self.mpcHorizon):
            objective += cv.quad_form(self.x[:,k] - xr, self.Q) + cv.quad_form(self.u[:,k], self.R)
            constraints += [self.x[:,k+1] == self.A@self.x[:,k] + self.B@(self.u[:,k] + np.array([self.m*self.g, 0, 0, 0])) + self.Bg]
            constraints += [self.xmin <= self.x[:,k], self.x[:,k] <= self.xmax]
            constraints += [self.umin <= self.u[:,k], self.u[:,k] <= self.umax]
        objective += cv.quad_form(self.x[:,self.mpcHorizon] - xr, self.Uinf)
        prob = cv.Problem(cv.Minimize(objective), constraints)

        return prob

    def state_update(self, odomInput):
        """ Generate state vector from odometry input"""
        # create state vector
        state = np.zeros((12))        
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

    def calc_ref_state(self, currTime):
        """ Function to calculate the reference state given the current time"""
        # find the closest index in timeVec corresponding to the current time        
        nearestIdx = np.searchsorted(self.timeVec, currTime)

        if nearestIdx == 0:
            nearestIdx = 1
        elif nearestIdx >= np.size(self.timeVec):
            nearestIdx = np.size(self.timeVec)-1 
        
        refState = np.array([self.waypoints[nearestIdx,0],
                            self.waypoints[nearestIdx,1],
                            self.waypoints[nearestIdx,2],
                            0,
                            0,
                            0,
                            0,
                            0,
                            self.waypoints[nearestIdx,3],
                            0,
                            0,
                            0])
        return refState

    def ctrl_update(self, state):
        """ Perform optimization and then formulate motor speeds"""
        xInit = cv.Parameter(self.nx)
        xInit.value = state

        # calculate the time difference
        # time now subtracted by start time
        currTime = rospy.get_time() - self.startTime
        xr = self.calc_ref_state(currTime)
        prob = self.mpc_problem_def(state, xr)
        prob.solve(solver=cv.OSQP, warm_start=True, verbose=False)
        desiredInput = self.u[:,0].value + self.equilibriumInput
        desiredInput = desiredInput.reshape(desiredInput.shape[0],-1)
        # find the rotor speed for each rotor
        motorSpeeds = Actuators()                
        motorSpeeds.angular_velocities = np.zeros((4,1))
        motorSpeedTransitionVec = np.dot(np.linalg.inv(self.speedAllocationMatrix), desiredInput)
        motorSpeeds.angular_velocities = np.sqrt(np.abs(motorSpeedTransitionVec))

        self.mpcPublisher.publish(motorSpeeds)
    
    def mpc_converter(self):
        """ Subscribe to the estimator """
        # rospy.Subscriber("/hummingbird/ground_truth/odometry", Odometry, self.state_update, queue_size = 1)
        rospy.Subscriber("/localization/odom", Odometry, self.state_update, queue_size = 1)
        rospy.spin()

def main():
    rospy.init_node("mpc_node", anonymous = False)
    mpcOperator = MPCQuadProg()

    try:
        mpcOperator.mpc_converter()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()







