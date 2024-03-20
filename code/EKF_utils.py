# utilities for EKF functions
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from pr3_utils import *
from scipy.sparse import csr_matrix, lil_matrix
from tqdm import tqdm
import pdb

def IMU_localization(linear_velocity,angular_velocity,time):
    omega_hat = hat_map(angular_velocity.T)
    zeta_hat = generate_hat_map_matrices(omega_hat,linear_velocity.T)
    # drop first pose
    zeta_hat = zeta_hat[1:]
    T_k = np.eye(4)
    T_history = []
    time_reshaped  = time[:, np.newaxis, np.newaxis]
    exp_map = expm(time_reshaped*zeta_hat) 
    for i in range(zeta_hat.shape[0]):
        T_k1 =  T_k @ exp_map[i]
        # print("localization",np.max(abs(T_k1)))
        # update T_k
        T_k = T_k1
        T_history.append(T_k1)
    return T_history

def visualize_trajectory(pose,dataset,save=False):
    fig,ax = plt.subplots(figsize=(8,6))
    x,y = transform_pose_matrix_to_xy(np.array(pose))
    ax.scatter(x[0],y[0],marker='s',label="start")
    ax.scatter(x[-1],y[-1],marker='o',label="end")
    plt.title(f"IMU localization via EKF prediction dataset {dataset}")
    plt.plot(x,y,label="IMU Localization via EKF Prediction")
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend()
    plt.grid()
    if save == True:
        plt.savefig(f'results/IMU_localization_via_EKF_prediction_dataset{dataset}.png')
    plt.show()
def visualize_landmark_mapping(lm, x, y,dataset,save=False,outlier_rejection=False):
    if outlier_rejection == True:
      if dataset == "10":
          x_mask = (lm[0, :] > -1500) & (lm[0, :] < 500)
          y_mask = (lm[1, :] > -1000) & (lm[1, :] < 500)
      elif dataset == "03":
          x_mask = (lm[0, :] > -1100) & (lm[0, :] < 500)
          y_mask = (lm[1, :] > -300) & (lm[1, :] < 700)
      mask = x_mask & y_mask
      lm = lm[:, mask]
    fig,ax = plt.subplots(figsize=(8,6))
    ax.scatter(x[0],y[0],marker='s',label="start")
    ax.scatter(x[-1],y[-1],marker='o',label="end")
    plt.title("Landmark Mapping with EKF Update")
    plt.plot(lm[0,:],lm[1,:],'.k',markersize=1,label="Updated landmarks")
    plt.plot(x, y, color='lime', label="robot trajectory",linewidth=2)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend()
    if save == True:
    #   plt.savefig(f"test_noise/landmark_EKF_{dataset}_noise_V1e-4_cov13-3.png")
      plt.savefig(f"results/SLAM_EKF_{dataset}.png")
    plt.show()

def flip_velocity_and_angular(velocity,angular):
    # Filp the y and z axis of the velocity and angular
    velocity[1] = -velocity[1]
    velocity[2] = -velocity[2]
    angular[1] = -angular[1]
    angular[2] = -angular[2]
    return velocity,angular

def homogenous(vectors_3d):
    '''
    @Input:
        vectors_3d = M x n = M elements of n 3d vectors
    '''
    return np.vstack((vectors_3d, np.ones((1, vectors_3d.shape[1]))))

def landmark_initialization(features,POSE,imu_T_cam,K_s,dataset,outlier_rejection=False):
    # x_all = []
    # y_all = []
    obs_features = []
    m_all = []
    observed = np.zeros(features.shape[1])
    landmark = np.ones((3, features.shape[1])) * -1
    for i in range(features.shape[2]):
        features_t = features[:,:,i]
        # index of useful features
        index = np.where(np.min(features_t, axis=0) != -1)
        # index of previous unobserved landmark 
        index_unobserved = np.where(observed == 0)
        index_observed = np.where(observed == 1)
        unobserved = np.intersect1d(index, index_unobserved)
        obs = np.intersect1d(index, index_observed)
        z_t = features_t[:, unobserved]
        obs_features.append(features_t[:, obs])
        # calculate the disparity between left to the right: uL - uR = 1/z * fx b
        disparity = z_t[0, :] - z_t[2, :]
        fu_b = - K_s[2, -1]
        # calculate the depth estimation of the 3D points z = fxb/uL-uR
        z = fu_b / disparity
        zt_ = np.ones((4, z_t.shape[1]))
        zt_[2, :] = z
        # x = uL * z / fx,  y = uR * z / fy
        zt_[0, :] = (z_t[0, :] * z - K_s[0,2] * z) / K_s[0,0]
        zt_[1, :] = (z_t[1, :] * z - K_s[1,2] * z) / K_s[1,1]
        # Z in camera frame
        z_r = imu_T_cam @ zt_
        m_w = POSE[i] @ z_r
        observed[unobserved] = 1
        landmark[:, unobserved] = m_w[:3, :]/m_w[-1, :]
        print("landmark",np.max(abs(landmark)))
        m_all.append(m_w[:3, :]/m_w[-1, :])
        # x_all.append(m_w[0,:])
        # y_all.append(m_w[1,:])

    if outlier_rejection == True:
      if dataset == "10":
          x_mask = (landmark[0, :] > -1500) & (landmark[0, :] < 500)
          y_mask = (landmark[1, :] > -1000) & (landmark[1, :] < 500)
      elif dataset == "03":
          x_mask = (landmark[0, :] > -1100) & (landmark[0, :] < 500)
          y_mask = (landmark[1, :] > -300) & (landmark[1, :] < 700)
      mask = x_mask & y_mask
      landmark = landmark[:, mask]
    return landmark, m_all, obs_features

def EKF_predicition_covariance(zeta,time):
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
    return Covariance

def generate_T_imu2o(imu_T_cam):
    o_T_r = np.array([[0,-1,0,0],
                      [0,0,-1,0],
                      [1,0, 0,0],
                      [0,0, 0,1]])
    o_T_i = o_T_r @ np.linalg.inv(imu_T_cam)
    return o_T_i

def compute_H(K_s,o_T_w,mean):
    # in part b assuming N = M
    N = mean.shape[1]
    M = N
    P = np.block([[np.eye(3), np.zeros((3, 1))]])
    H = np.zeros((4*N, 3*M))
    for i in range(N):
        H[4*i:4*i+4, 3*i:3*i+3] = K_s @ projectionJacobian((o_T_w @ mean[:,i]).reshape(1,-1)) @ o_T_w @ P.T
    return H

def compute_kalman_gain(sigma, H):
    measurement_noise = 5 * np.eye(H.shape[0])
    # try pinv if singular
    K = sigma @ H.T @ np.linalg.pinv(H @ sigma @ H.T + measurement_noise)
    return K

def compute_new_covariance(sigma, K, H):
    I = np.eye(sigma.shape[0])
    new_sigma = (I - K @ H) @ sigma
    return new_sigma

def EKF_update(features,lm,POSE,o_T_i,K_S):
    # EKF update  	
	observed = np.zeros(features.shape[1])
	# covariance 3M x 3M
	cov_sigma = 1e-3
	covariance = np.eye(features.shape[1] * 3) * cov_sigma
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
			o_T_w = o_T_i @ inversePose(POSE[i+1]) 
			# H: 4N x 3M (now 4N x 3N)
			H = compute_H(K_S,o_T_w,homogenous(mean))
			# kalman gain 3M x 4N (now 3N x 4N)
			K_gain = compute_kalman_gain(cov,H)
			I = np.eye(len(obs) * 3)
			# update mean, mean : 3Mx1
			mean = mean.T.reshape(-1,1) + K_gain @ inovation.T.reshape(-1,1)
			# update landmarks, convert mean back (need to follow the order of the reshape originally) 
			lm[:,obs] = mean.reshape(-1, 3).T
			# print("updated landmarks",np.max(abs(K_gain @ inovation.T.reshape(-1,1))))
            # update covariance, K_gain @ H : 3M x 3M
			# update covariance, K_gain @ H : 3M x 3M
			cov = (I - K_gain @ H) @ cov
			covariance[cov_mask_x, cov_mask_y] = cov

		# update observed landmarks indices
		observed[unobserved] = 1
		# pdb.set_trace()
	return lm

def EKF_predict(velocity,angular_velocity,time,T_k,cov_t):
    """
    EKF Prediction step, compute the mean and covariance of the predicted state
    @ Input:
    velocity: 3 x 1
    angular_velocity: 3 x 1
    time: 1 x 1
    @ output:
    T_k1: 4 x 4
    cov_t1: 6 x 6
    """
    omega_hat = hat_map(angular_velocity.T)
    zeta_hat = generate_hat_map_matrices(omega_hat,velocity.T)[0]
    exp_map = expm(time*zeta_hat)
    # try flipping the order
    T_k1 = T_k @ exp_map
    # compute covariance
    zeta = np.concatenate((velocity, angular_velocity), axis=0).T
    zeta_ad = axangle2adtwist(zeta)[0]
    W = 1e-3
    cov_t1 = expm(-time*zeta_ad) @ cov_t @ expm(-time*zeta_ad).T + np.eye(6) * W
    return T_k1, cov_t1

def lm_init(features_t,unobserved,pose,imu_T_cam,K_s):
    # initialize landmarks
    z_t = features_t[:, unobserved]
    # calculate the disparity between left to the right: uL - uR = 1/z * fx b
    disparity = z_t[0, :] - z_t[2, :]
    fu_b = - K_s[2, -1]
    # calculate the depth estimation of the 3D points z = fxb/uL-uR
    z = fu_b / disparity
    zt_ = np.ones((4, z_t.shape[1]))
    zt_[2, :] = z
    # x = uL * z / fx | y = uR * z / fy
    zt_[0, :] = (z_t[0, :] * z - K_s[0,2] * z) / K_s[0,0]
    zt_[1, :] = (z_t[1, :] * z - K_s[1,2] * z) / K_s[1,1]
    # Z in camera frame
    z_r = imu_T_cam @ zt_
    m_w = pose @ z_r
    return m_w[:3, :]/m_w[-1, :]

def Visual_SLAM(features, linear_velocity, angular_velocity, time, K_s, imu_T_cam):
    '''
    EKF Localization and Mapping.
    '''
    # set initial pose to be identity
    pose = np.eye(4)
    # o_T_i = generate_T_imu2o(imu_T_cam)
    o_T_i = np.linalg.inv(imu_T_cam)
    observed = np.zeros(features.shape[1])
    landmark = np.ones((3, features.shape[1])) * -1
    # covariance 3M x 3M
    cov_sigma = 1e-3
    pose_cov = np.diag([1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4])
    covariance_lm = (np.eye(features.shape[1] * 3) * cov_sigma)
    lm_pose_correlation = np.zeros((features.shape[1] * 3, 6))
    index_six = np.array([0,1,2,3,4,5])
    updated_pose_history = []
    for i in tqdm(range(features.shape[2]-1)):
        # EKF Prediction
        # print("updated pose",np.max(abs(pose)))
        pose, pose_cov = EKF_predict(linear_velocity[:,i].reshape(-1,1),angular_velocity[:,i].reshape(-1,1),time[i], pose, pose_cov)
        # print("predicted pose",np.max(abs(pose)))
        # pdb.set_trace()
        features_t = features[:,:,i]
        # index of useful features
        index = np.where(np.min(features_t, axis=0) != -1)
        # index of previous unobserved landmark 
        index_unobserved = np.where(observed == 0)
        index_observed = np.where(observed == 1)
        unobserved = np.intersect1d(index, index_unobserved)
        obs = np.intersect1d(index, index_observed)
        ind = np.union1d(obs, np.setdiff1d(index_observed,obs)[-100:])
        # landmark initial guess
        try:
            landmark[:, unobserved] = lm_init(features_t, unobserved, pose, imu_T_cam, K_s)
        except ValueError:
            print("ValueError in landmark initialization")
            pdb.set_trace()
        if len(obs) > 0:
            N, M = len(obs), len(ind)
            mean = landmark[:,ind]
            obs_z = features[:,:,i][:,obs]
            # 3M x 3M
            cov_mask = np.concatenate([ind * 3, ind * 3 + 1, ind * 3 + 2])
            cov_mask_x, cov_mask_y = np.meshgrid(cov_mask, cov_mask, sparse=False, indexing='xy')
            cov_lm = covariance_lm[cov_mask_x, cov_mask_y]
            # 6 x 3M
            corr_mask = np.concatenate([ind * 3, ind * 3 + 1, ind * 3 + 2])
            corr_mask_x, corr_mask_y = np.meshgrid(corr_mask, index_six, sparse=False, indexing='xy')
            cov_lm_pose_correlation = lm_pose_correlation[corr_mask_x, corr_mask_y].T
            z_o = o_T_i @ inversePose(pose) @ homogenous(landmark[:,obs])
            z_est = K_s @ (z_o/z_o[2,:])
            # z - z_est
            inovation = obs_z - z_est
            # inovation = (obs_z.T.reshape(-1, 1) - z_est.T.reshape(-1, 1)).flatten()
            # T from world to optical frame inversePose
            o_T_w = o_T_i @ inversePose(pose) 
            # o_T_w = o_T_i @ np.linalg.inv(pose) 
            # H: 4 N x (3 M + 6), Each H is 4 x 3
            H = np.zeros((4 * N, 3 * M + 6))
            P = np.block([[np.eye(3), np.zeros((3, 1))]])
            mean_h = homogenous(landmark[:,obs])
            for j in range(N):
                k = np.where(ind == obs[j])[0][0]
                H[4*j:4*j+4, 3*k:3*k+3] = (K_s @ projectionJacobian((o_T_w @ mean_h[:,j]).reshape(1,-1)) @ o_T_w @ P.T)[0]
                H[4*j:4*j+4, -6:] = -(K_s @ projectionJacobian((o_T_w @ mean_h[:,j]).reshape(1,-1)) @ o_T_i @ o_dot((inversePose(pose) @ mean_h[:,j]).reshape(1,-1)))[0]
            # pdb.set_trace()
            cov_lm_pose = np.zeros((3 * M + 6, 3 * M + 6)) 
            cov_lm_pose[:-6, -6:] = cov_lm_pose_correlation
            cov_lm_pose[-6:, :-6] = cov_lm_pose_correlation.T
            cov_lm_pose[-6:, -6:] = pose_cov
            cov_lm_pose[:-6, :-6] = cov_lm
            try:
                K_GAIN = cov_lm_pose @ H.T @ np.linalg.inv(H @ cov_lm_pose @ H.T + 1 * np.eye(H.shape[0])) #50
            except np.linalg.LinAlgError: 
                print("LinAlgError")
                pdb.set_trace()
            # update pose
            # pdb.set_trace()
            pose = pose @ expm(SE3_skew((K_GAIN[-6:, :] @ inovation.T.reshape(-1,1)).flatten()))
            # pdb.set_trace()
            updated_pose_history.append(pose)
            # update covariance jointly
            joint_cov = (np.eye(3 * M + 6) - K_GAIN @ H) @ cov_lm_pose
            pose_cov = joint_cov[-6:, -6:]
            cov_lm = joint_cov[:-6, :-6]
            cov_lm_pose_correlation = joint_cov[:-6, -6:]
            # update pose covariance
            # pose_cov = (np.eye(6) - K_GAIN[-6:, :] @ H[:, -6:]) @ pose_cov
            # update landmarks, mean = 3M x 1
            mean = mean.T.reshape(-1,1) + K_GAIN[:-6, :] @ inovation.T.reshape(-1,1)
            landmark[:,ind] = mean.reshape(-1, 3).T
            # update landmarks covariance
            # cov_lm = (np.eye(3 * M) - K_GAIN[:-6,:] @ H[:, :-6]) @ cov_lm
            covariance_lm[cov_mask_x, cov_mask_y] = cov_lm
            lm_pose_correlation[corr_mask_x, corr_mask_y] = cov_lm_pose_correlation.T
        observed[unobserved] = 1

    return updated_pose_history, landmark
    # N, M = obs.shape[1], len(index)
    
if __name__ == "__main__":
    print("This is a library of utility functions for EKF.")