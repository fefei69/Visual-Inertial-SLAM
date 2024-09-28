import numpy as np
from pr3_utils import *
from EKF_utils import *
import argparse
if __name__ == '__main__':
	# arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", type=str, default="03", help="dataset number 03, 10")
	parser.add_argument("--FLIP_IMU", action='store_true', help="Flip IMU")
	parser.add_argument("--Landmark_Mapping", action='store_true', help="Visualize Landmark Mapping or Visual SLAM")
	args = parser.parse_args()

	# Load the measurements
	filename = f"../data/{args.dataset}.npz"
	t, features, linear_velocity, angular_velocity, K, b, imu_T_cam = load_data(filename)

	FLIP_IMU = args.FLIP_IMU
	Visualize_Landmark_Mapping = args.Landmark_Mapping

	import pdb; pdb.set_trace()
	# compute tau
	time = t[0][1:] - t[0][0:-1]
	o_T_i = generate_T_imu2o(imu_T_cam)
	# Extrinsics
	K_S = np.block([[K[:2, :], np.array([[0, 0]]).T], 
				    [K[:2, :], np.array([[-K[0, 0] * b, 0]]).T]])
	# visualize_trajectory(POSE,dataset=dataset,save=False)
	if FLIP_IMU == True:
		linear_velocity, angular_velocity = flip_velocity_and_angular(linear_velocity, angular_velocity)
	# IMU Localization via EKF Prediction
	POSE = IMU_localization(linear_velocity, angular_velocity, time)
	# add identity pose to the beginning 
	POSE.insert(0,np.eye(4))
	# visualize localization
	visualize_trajectory(POSE,dataset=args.dataset,save=False)
	# IMU Localization via EKF Prediction
	zeta = np.concatenate((linear_velocity, angular_velocity), axis=0).T
	# EKF
	if Visualize_Landmark_Mapping == True:
		# covariance prediction 
		pred_cov = EKF_predicition_covariance(zeta, time)
		lm, m_bar, observed_features = landmark_initialization(features,POSE,imu_T_cam,K_S,args.dataset,outlier_rejection=False)
		x, y = transform_pose_matrix_to_xy(np.array(POSE))
		visualize_landmark_mapping(lm, x, y,args.dataset,save=False,outlier_rejection=False)
		# EKF update
		landmark_test = EKF_update(features, lm, POSE, o_T_i, K_S)
		# np.save(f"results/{dataset}_landmarks_m_noise{cov_sigma}.npy",lm)
		visualize_landmark_mapping(landmark_test, x, y,args.dataset,save=False,outlier_rejection=True)
	else:
		slam_poses, landmarks = Visual_SLAM(features, linear_velocity, linear_velocity, time, K_S, imu_T_cam)
		x, y = transform_pose_matrix_to_xy(np.array(slam_poses))
		visualize_landmark_mapping(landmarks, x, y,args.dataset,save=True,outlier_rejection=True)









