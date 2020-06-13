# Quadcopter Control
Testing out different control laws for quadcopter control, currently have infinite discrete linear quadratic regulator implemented. Using rotorS from ETH-Zurich as simulation environment which can be accessed here: https://github.com/ethz-asl/rotors_simulator.

Development work is done under the quad_adcs folder.

Not really sure how to get `catkin_make` working with the `catkin build` from **rotors_simulator**, so the control code is separated.

## Steps to Run Simulation
Launch the simulation environment
* $ roslaunch rotors_gazebo mav.launch mav_name:=hummingbird world_name:=basic

Launch the controller inside the quad_adcs/src folder (need to figure out how to get rosrun working with catkin_build)
* /catkin_ws/src/quad_adcs/src (master) $ python discrete_lqr.py 

