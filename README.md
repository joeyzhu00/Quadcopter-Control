# Quadcopter Control
Testing out different control laws for quadcopter control, currently have infinite discrete linear quadratic regulator implemented in addition to a minimum jerk trajectory generator. Using rotorS from ETH-Zurich as simulation environment which can be accessed here: https://github.com/ethz-asl/rotors_simulator.

Development work is done under the quad_adcs folder.

Not really sure how to get `catkin_make` working with the `catkin build` from **rotors_simulator**, so the control code is separated.

## Steps to Run Simulation
Launch the simulation environment
* $ roslaunch rotors_gazebo mav.launch mav_name:=hummingbird world_name:=basic

Launch the controller inside the quad_adcs/src folder (need to figure out how to get rosrun working with catkin_build)
* /catkin_ws/src/quad_adcs/src (master) $ python discrete_lqr.py

## Sub-Optimal Minimum Jerk Trajectory
Calling the trajectory generator "sub-optimal" due to using pseudoinverse to generate trajectory generator coefficients for the 0<sup>th</sup> to (n-1)<sup>th</sup> points. The last two waypoints exhibit an actual minimum jerk trajectory which can be observed by the kink in the yaw acceleration trajectory.

![sub_optimal_min_jerk_waypoints](https://user-images.githubusercontent.com/29212589/85236328-eaf9d600-b3d1-11ea-96fd-4626667bda51.png)


![sub_optimal_min_jerk_desired_kinematics](https://user-images.githubusercontent.com/29212589/85236330-ecc39980-b3d1-11ea-86bb-6faa3d147d89.png)

