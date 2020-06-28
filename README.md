# Quadcopter Control
Testing out different control laws for quadcopter control, currently have infinite discrete linear quadratic regulator and PD controller implemented in addition to a minimum jerk trajectory generator. Using rotorS from ETH-Zurich as simulation environment which can be accessed here: https://github.com/ethz-asl/rotors_simulator.

Development work is done under the quad_adcs folder.

Not really sure how to get `catkin_make` working with the `catkin build` from **rotors_simulator**, so the control code is separated.

## Steps to Run Simulation
Launch the simulation environment (wherever you keep rotorS)
* $ roslaunch rotors_gazebo mav.launch mav_name:=hummingbird world_name:=basic

# DLQR
* /quad_ws (master) $ source devel/setup.bash
* /quad_ws (master) $ rosrun control discrete_lqr.py
# PD Control
* /quad_ws (master) $ source devel/setup.bash
* /quad_ws (master) $ rosrun control pd_control.py

## Infinite Horizon Discrete LQR Performance
Sluggish yaw response, but reasonably quick position convergence. Can probably use more time for tuning gains to get rid of the overshoot and get more of a critically damped response. The plot is with live data with a one second interval represented by symbols (not sure why two of each appear in the legends). Need to tune out steady-state instability that occurs about 20 seconds after reaching the desired "steady-state position". May keep this as a feature as the quadcopter does some complex acrobatics during this unstable phase. Possible limit cycle occurring in the phase plane of the quadcopter that is causing this instability.  

![dlqr_step_response](https://user-images.githubusercontent.com/29212589/85929675-9fc94280-b86b-11ea-91a3-cc68ee14d815.png)

## PD Control Performance
Roll/Pitch PD gains based off assumption that the quadcopter can be approximated as a second-order system. Small yaw torque authority results in the addition of a multiplier for the yaw PD gains. Position control gains are hand-tuned. Roll/Pitch damping ratio is overdamped while yaw is critically damped.

![pd_control_step_response](https://user-images.githubusercontent.com/29212589/85929927-57128900-b86d-11ea-81c3-26af6c4765b1.png)

## Sub-Optimal Minimum Jerk Trajectory
Calling the trajectory generator "sub-optimal" due to using pseudoinverse to generate trajectory generator coefficients for the 0<sup>th</sup> to (n-1)<sup>th</sup> points. The last two waypoints exhibit an actual minimum jerk trajectory which can be observed by the kink in the yaw acceleration trajectory.

![sub_optimal_min_jerk_waypoints](https://user-images.githubusercontent.com/29212589/85929683-ac4d9b00-b86b-11ea-825d-3083b69145da.png)

![sub_optimal_min_jerk_desired_kinematics](https://user-images.githubusercontent.com/29212589/85929684-ad7ec800-b86b-11ea-9e30-e5e20ba5fb86.png)
