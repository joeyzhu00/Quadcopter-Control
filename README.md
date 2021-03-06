# Quadcopter Control
Testing out different control laws for quadcopter control, currently have the following features:
* finite horizon discrete linear quadratic regulator
* infinite horizon discrete linear quadratic regulator
* infinite horizon discrete linear quadratic regulator with integral action
* PD controller
* MPC with linear quadratic programming (yaw authority basically nonexistent, needs tuning)
* minimum jerk trajectory generator 
* EKF subscribed to rotorS sensors

Using rotorS from ETH-Zurich as the simulation environment which can be accessed here: https://github.com/ethz-asl/rotors_simulator.
## Installing Minimum Version of rotors_simulator
```
$ mkdir -p ~/catkin_ws/src
$ cd ~/catkin_ws/src
$ catkin_init_workspace  # initialize your catkin workspace
$ wstool init
$ wget https://raw.githubusercontent.com/ethz-asl/rotors_simulator/master/rotors_minimal.rosinstall
$ wstool merge rotors_hil.rosinstall
$ wstool update
$ cd ~/catkin_ws/
$ catkin build
$ echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
$ source ~/.bashrc
```
Not really sure how to get `catkin_make` working with the `catkin build` from **rotors_simulator**, so the control code is separated.

**NOTE:** The infinite horizon discrete LQR methods are linearized about a single equilibrium point, so large yaw angles will cause instability due to small angle assumptions causing "unmodeled dynamics" to arise. LQR methods will probably need a camera gimbal deal with the yaw pointing.  

## Steps to Run Simulation
Open up `waypoint_generation_library.py` and check whether the waypoints hardcoded in are what you want (eventually this will be part of a launch or YAML file). 

Launch the simulation environment (wherever you keep rotorS)
* $ roslaunch rotors_gazebo mav.launch mav_name:=hummingbird world_name:=basic

**NOTE:** The following launch files include the EKF.

# Infinite Horizon DLQR
```
/Quadcopter-Control (master) $ source devel/setup.bash
/Quadcopter-Control (master) $ roslaunch control simulation_inf_lqr.launch
```
# Infinite Horizon DLQR With Integral Action
```
/Quadcopter-Control (master) $ source devel/setup.bash
/Quadcopter-Control (master) $ roslaunch control simulation_inf_lqr_with_integral.launch
```
# PD Control
```
/Quadcopter-Control (master) $ source devel/setup.bash
/Quadcopter-Control (master) $ roslaunch control simulation_pd_control.launch
```
# MPC
```
/Quadcopter-Control (master) $ source devel/setup.bash
/Quadcopter-Control (master) $ roslaunch control simulation_mpc.launch
```

## Infinite Horizon Discrete LQR Performance
Sluggish yaw response, but reasonably quick position convergence. Can probably use more time for tuning gains to get rid of the overshoot and get more of a critically damped response. The plot is with live data with a one second interval represented by symbols (not sure why two of each appear in the legends).   

![dlqr_step_response](https://user-images.githubusercontent.com/29212589/85929675-9fc94280-b86b-11ea-91a3-cc68ee14d815.png)

## PD Control Performance
Roll/Pitch PD gains based off assumption that the quadcopter can be approximated as a second-order system. Small yaw torque authority results in the addition of a multiplier for the yaw PD gains. Position control gains are hand-tuned. Roll/Pitch damping ratio is overdamped while yaw is critically damped.

![pd_control_step_response](https://user-images.githubusercontent.com/29212589/85929927-57128900-b86d-11ea-81c3-26af6c4765b1.png)

## MPC - Quadratic Program 
Linear quadratic program with gains from the infinite horizon discrete LQR controller. XYZ-position control is available but the yaw control needs a lot of work. It is very sensitive to the MPC horizon, high computation speed is a must when using the MPC. 

## Optimal Minimum Jerk Trajectory
To get intermediate velocity and acceleration targets between the desired waypoints, a pseudo-inverse solution gets computed. Afterwards, a minimum jerk trajectory between each desired waypoint is generated. 

![Minimum Jerk Waypoints](https://user-images.githubusercontent.com/29212589/87103039-b6c34980-c208-11ea-8db5-8807edf11a69.png)

![Desired Kinematics](https://user-images.githubusercontent.com/29212589/87103018-a3b07980-c208-11ea-9ca5-609c0b0d52f8.png)

## Single Waypoint With Finite Horizon Discrete LQR
![quadcopter_single_waypoint](https://user-images.githubusercontent.com/29212589/90966308-f3c86000-e485-11ea-9282-9bfe9f907583.gif)
