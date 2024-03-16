import numpy as np
from pr3_utils import *
import pdb
from tqdm import tqdm
from scipy.sparse import csr_matrix, lil_matrix
if __name__ == '__main__':

	# Load the measurements
	dataset = "03"
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
	lm, m_bar, observed_features = landmark_initialization(features,POSE,imu_T_cam,K_S,dataset,outlier_rejection=False)
	if Visualize_Landmark_Mapping == True:
		x, y = transform_pose_matrix_to_xy(np.array(POSE))
		visualize_landmark_mapping(lm, x, y,dataset,save=False,outlier_rejection=False)

	# EKF update  	
	observed = np.zeros(features.shape[1])
	# covariance 3M x 3M
	cov_sigma = 1e-3
	covariance = lil_matrix(np.eye(features.shape[1] * 3) * cov_sigma)
	landmark_test = np.ones((3, features.shape[1])) * -1
	# covariance = np.eye(1000 * 3) * 0.1
	for i in tqdm(range(features.shape[2] - 1)):
		# index of useful features
		index = np.where(np.min(features[:,:,i], axis=0) != -1)
		# index of previous unobserved landmark 
		index_unobserved = np.where(observed == 0)
		index_observed = np.where(observed == 1)
		unobserved = np.intersect1d(index, index_unobserved)
		obs = np.intersect1d(index, index_observed)
		ind = np.union1d(obs, np.setdiff1d(index_observed,obs)[-100:])
		# new observation
		if len(obs) > 0:
			cov_mask = np.concatenate([obs * 3, obs * 3 + 1, obs * 3 + 2])
			cov_mask_x, cov_mask_y = np.meshgrid(cov_mask, cov_mask, sparse=False, indexing='xy')
			cov = covariance[cov_mask_x, cov_mask_y]	
			obs_z = features[:,:,i+1][:,obs]
			mean = lm[:,obs]
			z_o = o_T_i @ np.linalg.inv(POSE[i+1]) @ homogenous(lm[:,obs])
			z_est = K_S @ (z_o/z_o[2,:])
			# z - z_est
			inovation = obs_z - z_est
			# T from world to optical frame
			o_T_w = o_T_i @ np.linalg.inv(POSE[i+1]) 
			# H: 4N x 3M (now 4N x 3N)
			H = compute_H(K_S,o_T_w,homogenous(mean))
			# kalman gain 3M x 4N (now 3N x 4N)
			K_gain = compute_kalman_gain(cov,H)
			I = np.eye(len(obs) * 3)
			# update mean, mean : 3Mx1
			mean = mean.T.reshape(-1,1) + K_gain @ inovation.T.reshape(-1,1)
			# update landmarks, convert mean back (need to follow the order of the reshape originally) 
			landmark_test[:,obs] = mean.reshape(-1, 3).T
			# pdb.set_trace()
			# update covariance, K_gain @ H : 3M x 3M
			cov = (I - K_gain @ H) @ cov
			covariance[cov_mask_x, cov_mask_y] = cov

		# update observed landmarks indices
		observed[unobserved] = 1
		# pdb.set_trace()
	x, y = transform_pose_matrix_to_xy(np.array(POSE))
	# np.save(f"results/{dataset}_landmarks_m_noise{cov_sigma}.npy",lm)
	visualize_landmark_mapping(landmark_test, x, y,dataset,save=True,outlier_rejection=True)















	# (b) Landmark Mapping via EKF Update
	# (c) Visual-Inertial SLAM
	# You may use the function below to visualize the robot pose over time
	# visualize_trajectory_2d(world_T_imu, show_ori = True)


