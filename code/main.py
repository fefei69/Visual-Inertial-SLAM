import numpy as np
from pr3_utils import *
import pdb
from tqdm import tqdm
from scipy.sparse import csr_matrix, lil_matrix
from EKF_utils import *

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
	# IMU Localization via EKF Prediction
	zeta = np.concatenate((linear_velocity, angular_velocity), axis=0).T
	if Visualize_Landmark_Mapping == True:
		# prediction covariance
		pred_cov = EKF_predicition_covariance(zeta, time)
		lm, m_bar, observed_features = landmark_initialization(features,POSE,imu_T_cam,K_S,dataset,outlier_rejection=False)
		x, y = transform_pose_matrix_to_xy(np.array(POSE))
		visualize_landmark_mapping(lm, x, y,dataset,save=False,outlier_rejection=False)
		# EKF update
		landmark_test = EKF_update(features,lm,POSE,o_T_i,K_S)
		# np.save(f"results/{dataset}_landmarks_m_noise{cov_sigma}.npy",lm)
		visualize_landmark_mapping(landmark_test, x, y,dataset,save=True,outlier_rejection=True)
	else:
		Visual_SLAM(features, linear_velocity, linear_velocity, time, K_S,imu_T_cam)















	# (b) Landmark Mapping via EKF Update
	# (c) Visual-Inertial SLAM
	# You may use the function below to visualize the robot pose over time
	# visualize_trajectory_2d(world_T_imu, show_ori = True)


