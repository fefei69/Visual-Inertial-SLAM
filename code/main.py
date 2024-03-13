import numpy as np
from pr3_utils import *
import pdb
from tqdm import tqdm
if __name__ == '__main__':

	# Load the measurements
	dataset = "10"
	filename = f"../data/{dataset}.npz"
	t, features, linear_velocity, angular_velocity, K, b, imu_T_cam = load_data(filename)
	# compute tau
	time = t[0][1:] - t[0][0:-1]
	# IMU Localization via EKF Prediction
	POSE = IMU_localization(linear_velocity, angular_velocity, time)
	POSE.insert(0,np.eye(4))
	# visualize_trajectory(POSE,dataset=dataset,save=False)
	# Extrinsics
	K_S = np.block([[K[:2, :], np.array([[0, 0]]).T], [K[:2, :], np.array([[-K[0, 0] * b, 0]]).T]])
	# visualize_trajectory(POSE,dataset=dataset,save=False)
	FLIP = False
	if FLIP == True:
		linear_velocity, angular_velocity = flip_velocity_and_angular(linear_velocity, angular_velocity)
	# (a) IMU Localization via EKF Prediction
	# (6,n) -> (n,6)
	zeta = np.concatenate((linear_velocity, angular_velocity), axis=0).T
	# prediction covariance
	pred_cov = EKF_predicition_covariance(zeta, time)
	lm = landmark_initialization(features,POSE,imu_T_cam,K_S)
	x, y = transform_pose_matrix_to_xy(np.array(POSE))
	visualize_landmark_mapping(lm, x, y,dataset,save=False)
	
















	# (b) Landmark Mapping via EKF Update
	# (c) Visual-Inertial SLAM
	# You may use the function below to visualize the robot pose over time
	# visualize_trajectory_2d(world_T_imu, show_ori = True)


