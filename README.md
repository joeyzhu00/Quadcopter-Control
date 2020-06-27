# Quadcopter Control
Testing out different control laws for quadcopter control, currently have infinite discrete linear quadratic regulator implemented in addition to a minimum jerk trajectory generator. Using rotorS from ETH-Zurich as simulation environment which can be accessed here: https://github.com/ethz-asl/rotors_simulator.

Development work is done under the quad_adcs folder.

Not really sure how to get `catkin_make` working with the `catkin build` from **rotors_simulator**, so the control code is separated.

## Steps to Run Simulation
Launch the simulation environment
* $ roslaunch rotors_gazebo mav.launch mav_name:=hummingbird world_name:=basic

Launch the controller inside the quad_adcs/src folder (need to figure out how to get rosrun working with catkin_build)
* /catkin_ws/src/quad_adcs/src (master) $ python discrete_lqr.py

## Infinite Horizon Discrete LQR Performance
Sluggish yaw response, but reasonably quick position convergence. Can probably use more time for tuning gains to get rid of the overshoot and get more of a critically damped response. The plot is with live data with a one second interval represented by symbols (not sure why two of each appear in the legends). 

![dlqr_step_response](https://user-images.githubusercontent.com/29212589/85929675-9fc94280-b86b-11ea-91a3-cc68ee14d815.png)

## Sub-Optimal Minimum Jerk Trajectory
Calling the trajectory generator "sub-optimal" due to using pseudoinverse to generate trajectory generator coefficients for the 0<sup>th</sup> to (n-1)<sup>th</sup> points. The last two waypoints exhibit an actual minimum jerk trajectory which can be observed by the kink in the yaw acceleration trajectory.

![sub_optimal_min_jerk_waypoints](https://user-images.githubusercontent.com/29212589/85929683-ac4d9b00-b86b-11ea-825d-3083b69145da.png)

![sub_optimal_min_jerk_desired_kinematics](https://user-images.githubusercontent.com/29212589/85929684-ad7ec800-b86b-11ea-9e30-e5e20ba5fb86.png)
