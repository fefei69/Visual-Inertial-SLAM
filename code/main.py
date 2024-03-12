import numpy as np
from pr3_utils import *
import pdb

if __name__ == '__main__':

	# Load the measurements
	dataset = "10"
	filename = f"../data/{dataset}.npz"
	t, features, linear_velocity, angular_velocity, K, b, imu_T_cam = load_data(filename)
	time = t[0][1:] - t[0][0:-1]
	FLIP = False
	Sigma_initial = np.diag([1e-3, 1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4])
	pdb.set_trace()
	if FLIP == True:
		linear_velocity, angular_velocity = flip_velocity_and_angular(linear_velocity, angular_velocity)
	# (a) IMU Localization via EKF Prediction
	POSE = IMU_localization(linear_velocity, angular_velocity, time)
	# (6,n) -> (n,6)
	# zeta = np.concatenate((linear_velocity, angular_velocity), axis=0).reshape(-1, 6)
	# visualize_trajectory(POSE,dataset=dataset,save=False)


	# (b) Landmark Mapping via EKF Update

	# (c) Visual-Inertial SLAM

	# You may use the function below to visualize the robot pose over time
	# visualize_trajectory_2d(world_T_imu, show_ori = True)


