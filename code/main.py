import numpy as np
from pr3_utils import *
import pdb
from tqdm import tqdm
if __name__ == '__main__':

	# Load the measurements
	dataset = "10"
	filename = f"../data/{dataset}.npz"
	t, features, linear_velocity, angular_velocity, K, b, imu_T_cam = load_data(filename)
	Visualize_Landmark_Mapping = True
	# compute tau
	time = t[0][1:] - t[0][0:-1]
	# IMU Localization via EKF Prediction
	POSE = IMU_localization(linear_velocity, angular_velocity, time)
	POSE.insert(0,np.eye(4))
	o_T_i = generate_T_imu2o(imu_T_cam)
	# visualize_trajectory(POSE,dataset=dataset,save=False)
	# Extrinsics
	K_S = np.block([[K[:2, :], np.array([[0, 0]]).T], [K[:2, :], np.array([[-K[0, 0] * b, 0]]).T]])
	# visualize_trajectory(POSE,dataset=dataset,save=False)
	FLIP_IMU = False
	if FLIP_IMU == True:
		linear_velocity, angular_velocity = flip_velocity_and_angular(linear_velocity, angular_velocity)
	# (a) IMU Localization via EKF Prediction
	# (6,n) -> (n,6)
	zeta = np.concatenate((linear_velocity, angular_velocity), axis=0).T
	# prediction covariance
	pred_cov = EKF_predicition_covariance(zeta, time)
	lm, m_bar, observed_features = landmark_initialization(features,POSE,imu_T_cam,K_S)
	if Visualize_Landmark_Mapping == True:
		x, y = transform_pose_matrix_to_xy(np.array(POSE))
		visualize_landmark_mapping(lm, x, y,dataset,save=False)

	# EKF update  	
	observed = np.zeros(features.shape[1])
	for i in range(features.shape[2] - 1):
		# index of useful features
		index = np.where(np.min(features[:,:,i], axis=0) != -1)
		# index of previous unobserved landmark 
		index_unobserved = np.where(observed == 0)
		index_observed = np.where(observed == 1)
		unobserved = np.intersect1d(index, index_unobserved)
		obs = np.intersect1d(index, index_observed)
		ind = np.union1d(obs, np.setdiff1d(index_observed,obs)[-100:])
		mean = m_bar[i]
		# new observation
		if len(obs) > 0:
			obs_z = features[:,:,i+1][:,obs]
			try:
				z_o = o_T_i @ np.linalg.inv(POSE[i+1]) @ homogenous(lm[:,obs])
				z_est = K_S @ (z_o/z_o[2,:])
			except IndexError:
				print("Index Error")
				pdb.set_trace()
			try:
				# z - z_est
				inovation = obs_z - z_est
			except ValueError:
				print("Value Error")
				pdb.set_trace()
		# update observed landmarks indices
		observed[unobserved] = 1
		# pdb.set_trace()
















	# (b) Landmark Mapping via EKF Update
	# (c) Visual-Inertial SLAM
	# You may use the function below to visualize the robot pose over time
	# visualize_trajectory_2d(world_T_imu, show_ori = True)


