import numpy as np
from pr3_utils import *
import pdb
from tqdm import tqdm
if __name__ == '__main__':

	# Load the measurements
	dataset = "10"
	filename = f"../data/{dataset}.npz"
	t, features, linear_velocity, angular_velocity, K, b, imu_T_cam = load_data(filename)
	time = t[0][1:] - t[0][0:-1]
	POSE = IMU_localization(linear_velocity, angular_velocity, time)
	POSE.insert(0,np.eye(4))
	K_S = np.block([[K[:2, :], np.array([[0, 0]]).T], [K[:2, :], np.array([[-K[0, 0] * b, 0]]).T]])
	lm, m,x,y = landmark_initialization(features,POSE,imu_T_cam,K_S)
	x = []
	y = []
	for j in tqdm(range(len(m))):
		mw = m[j]
		for k in range(mw.shape[1]):
			x.append(mw[0, k])
			y.append(mw[1, k])
	x = np.array(x,dtype=np.float32)
	y = np.array(y,dtype=np.float32)
	plt.plot(lm[0,:],lm[1,:],'.g', markersize=1)
	plt.show()
	plt.plot(x, y, '.g', markersize=1)
	plt.show()
	pdb.set_trace()
	# visualize_trajectory(POSE,dataset=dataset,save=False)
	observations = []
	m_bar = []
	# relative transformation from left camera to right camera
	p = np.array([-b,0,0]).reshape(3,1)
	# p = np.array([0,b,0]).reshape(3,1)
	e_3 = np.array([0,0,1]).reshape(3,1)
	R = np.eye(3)
	o_T_r = np.array([[0,-1,0,0],
				      [0,0,-1,0],
				      [1,0, 0,0],
				      [0,0, 0,1]])
	o_T_i = o_T_r @ np.linalg.inv(imu_T_cam)
	x_all = []
	y_all = []
	# pdb.set_trace()
	for j in tqdm(range(1)):
	# for j in tqdm(range(features.shape[2])):
		# First element of each feature should not be -1
		valid = features[:,:,j][0]!=-1.
		features_t = features[:,:,j][:,valid]
		m_all = []
		# 3 x n
		z_1 = homogenous(np.stack((features_t[0],features_t[2])))
		z_2 = homogenous(np.stack((features_t[1],features_t[3])))
		# 3 x n
		a = R.T @ p - (e_3.T @ R.T @ p) * z_2
		b_ = R.T @ z_1 - (e_3.T @ R.T @ z_1) * z_2
		m_ = homogenous((np.dot(a,a.T)[0][0] / np.dot(a,b_.T)[0][0]) * z_1)
		m_hat = POSE[j] @ imu_T_cam @ m_
		x_all.append(m_hat[0,:])
		y_all.append(m_hat[1,:])
		pdb.set_trace()
	# 	for i in range(features_t.shape[1]):
	# 		# homogeneous
	# 		z_1 = np.append(np.asarray(features_t[:,i][0:2]),1).reshape(3,1)
	# 		z_2 = np.append(np.asarray(features_t[:,i][2:]),1).reshape(3,1)
	# 		a = R.T @ p - (e_3.T @ R.T @ p) * z_2
	# 		b_ = R.T @ z_1 - (e_3.T @ R.T @ z_1) * z_2
	# 		# homogeneous
	# 		m_ = np.append((a.T @ a / (a.T @ b_)) * z_1, 1).reshape(4,1) 
	# 		# pdb.set_trace()
	# 		# m_ = np.ones((4,1))
	# 		# m_hat = POSE[j] @ np.linalg.inv(imu_T_cam) @ m_
	# 		if -1000 < m_[0] < 1000:
	# 			m_hat = POSE[j] @ imu_T_cam @ m_
	# 			m_all.append(m_hat)
	# 			x_all.append(m_hat[0])
	# 			y_all.append(m_hat[1])
	# 		# z_t = K @ projection((o_T_i @ np.linalg.inv(POSE[j]) @ m_).T).T
	# 	m_all = np.array(m_all)
	# 	observations.append(m_all[...,0])
	# # observations = np.array(observations)
	plt.plot(x_all,y_all,'o')
	plt.show()
	pdb.set_trace()
	FLIP = False
	if FLIP == True:
		linear_velocity, angular_velocity = flip_velocity_and_angular(linear_velocity, angular_velocity)
	# (a) IMU Localization via EKF Prediction
	POSE = IMU_localization(linear_velocity, angular_velocity, time)
	# (6,n) -> (n,6)
	zeta = np.concatenate((linear_velocity, angular_velocity), axis=0).T
	zeta_ad = axangle2adtwist(zeta)
	# drop first pose
	zeta_ad = zeta_ad[1:]
	cov_t = np.diag([1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4])
	Covariance = [cov_t]
	W = 1e-3
	for i in range(zeta_ad.shape[0]):
		cov_t1 = expm(-time[i]*zeta_ad[i]) @ cov_t @ expm(-time[i]*zeta_ad[i]).T + W
		# update cov_t
		cov_t = cov_t1
		Covariance.append(cov_t1)
	pdb.set_trace()
	plt.plot(Covariance)
	plt.show()
	# visualize_trajectory(POSE,dataset=dataset,save=False)


	# (b) Landmark Mapping via EKF Update

	# (c) Visual-Inertial SLAM

	# You may use the function below to visualize the robot pose over time
	# visualize_trajectory_2d(world_T_imu, show_ori = True)


