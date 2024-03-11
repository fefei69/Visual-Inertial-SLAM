import numpy as np
from pr3_utils import *


if __name__ == '__main__':

	# Load the measurements
	filename = "../data/10.npz"
	t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)
	print("t: ", t)
	print("features: ", features)
	print("linear_velocity: ", linear_velocity)
	print("angular_velocity: ", angular_velocity)
	print("K: ", K)
	print("b: ", b)
	print("imu_T_cam: ", imu_T_cam)
	# (a) IMU Localization via EKF Prediction

	# (b) Landmark Mapping via EKF Update

	# (c) Visual-Inertial SLAM

	# You may use the function below to visualize the robot pose over time
	# visualize_trajectory_2d(world_T_imu, show_ori = True)


