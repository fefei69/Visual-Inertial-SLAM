import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler
from scipy.spatial.transform import Rotation
from scipy.linalg import expm
def load_data(file_name):
    '''
    function to read visual features, IMU measurements, and calibration parameters
    Input:
        file_name: the input data file. Should look like "XX.npz"
    Output:
        t: time stamp
            with shape 1*t
        features: visual feature point coordinates in stereo images, 
            with shape 4*n*t, where n is number of features
        linear_velocity: velocity measurements in IMU frame
            with shape 3*t
        angular_velocity: angular velocity measurements in IMU frame
            with shape 3*t
        K: (left)camera intrinsic matrix
            with shape 3*3
        b: stereo camera baseline
            with shape 1
        imu_T_cam: extrinsic transformation from (left) camera to imu frame, in SE(3).
            with shape 4*4
    '''
    with np.load(file_name) as data:
    
        t = data["time_stamps"] # time_stamps
        features = data["features"] # 4 x num_features : pixel coordinates of the visual features
        linear_velocity = data["linear_velocity"] # linear velocity in body-frame coordinates
        angular_velocity = data["angular_velocity"] # angular velocity in body-frame coordinates
        K = data["K"] # intrinsic calibration matrix
        b = data["b"] # baseline
        imu_T_cam = data["imu_T_cam"] # transformation from left camera frame to imu frame 
    
    return t,features,linear_velocity,angular_velocity,K,b,imu_T_cam


def visualize_trajectory_2d(pose,path_name="Unknown",show_ori=False):
    '''
    function to visualize the trajectory in 2D
    Input:
        pose:   4*4*N matrix representing the camera pose, 
                where N is the number of poses, and each
                4*4 matrix is in SE(3)
    '''
    fig,ax = plt.subplots(figsize=(5,5))
    n_pose = pose.shape[2]
    ax.plot(pose[0,3,:],pose[1,3,:],'r-',label=path_name)
    ax.scatter(pose[0,3,0],pose[1,3,0],marker='s',label="start")
    ax.scatter(pose[0,3,-1],pose[1,3,-1],marker='o',label="end")
  
    if show_ori:
        select_ori_index = list(range(0,n_pose,max(int(n_pose/50), 1)))
        yaw_list = []
        
        for i in select_ori_index:
            _,_,yaw = mat2euler(pose[:3,:3,i])
            yaw_list.append(yaw)
    
        dx = np.cos(yaw_list)
        dy = np.sin(yaw_list)
        dx,dy = [dx,dy]/np.sqrt(dx**2+dy**2)
        ax.quiver(pose[0,3,select_ori_index],pose[1,3,select_ori_index],dx,dy,\
            color="b",units="xy",width=1)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.grid(False)
    ax.legend()
    plt.show(block=True)

    return fig, ax




def projection(ph):
  '''
  ph = n x 4 = homogeneous point coordinates
  r = n x 4 = ph/ph[...,2] = normalized z axis coordinates
  '''  
  return ph/ph[...,2,None]
  
def projectionJacobian(ph):
  '''
  ph = n x 4 = homogeneous point coordinates
  J = n x 4 x 4 = Jacobian of ph/ph[...,2]
  '''  
  J = np.zeros(ph.shape+(4,))
  iph2 = 1.0/ph[...,2]
  ph2ph2 = ph[...,2]**2
  J[...,0,0], J[...,1,1],J[...,3,3] = iph2,iph2,iph2
  J[...,0,2] = -ph[...,0]/ph2ph2
  J[...,1,2] = -ph[...,1]/ph2ph2
  J[...,3,2] = -ph[...,3]/ph2ph2
  return J


def inversePose(T):
  '''
  @Input:
    T = n x 4 x 4 = n elements of SE(3)
  @Output:
    iT = n x 4 x 4 = inverse of T
  '''
  iT = np.empty_like(T)
  iT[...,0,0], iT[...,0,1], iT[...,0,2] = T[...,0,0], T[...,1,0], T[...,2,0] 
  iT[...,1,0], iT[...,1,1], iT[...,1,2] = T[...,0,1], T[...,1,1], T[...,2,1] 
  iT[...,2,0], iT[...,2,1], iT[...,2,2] = T[...,0,2], T[...,1,2], T[...,2,2]
  iT[...,:3,3] = -np.squeeze(iT[...,:3,:3] @ T[...,:3,3,None])
  iT[...,3,:] = T[...,3,:]
  return iT


def axangle2skew(a):
  '''
  converts an n x 3 axis-angle to an n x 3 x 3 skew symmetric matrix 
  '''
  S = np.empty(a.shape[:-1]+(3,3))
  S[...,0,0].fill(0)
  S[...,0,1] =-a[...,2]
  S[...,0,2] = a[...,1]
  S[...,1,0] = a[...,2]
  S[...,1,1].fill(0)
  S[...,1,2] =-a[...,0]
  S[...,2,0] =-a[...,1]
  S[...,2,1] = a[...,0]
  S[...,2,2].fill(0)
  return S

def axangle2twist(x):
  '''
  @Input:
    x = n x 6 = n elements of position and axis-angle
  @Output:
    T = n x 4 x 4 = n elements of se(3)
  '''
  T = np.zeros(x.shape[:-1]+(4,4))
  T[...,0,1] =-x[...,5]
  T[...,0,2] = x[...,4]
  T[...,0,3] = x[...,0]
  T[...,1,0] = x[...,5]
  T[...,1,2] =-x[...,3]
  T[...,1,3] = x[...,1]
  T[...,2,0] =-x[...,4]
  T[...,2,1] = x[...,3]
  T[...,2,3] = x[...,2]
  return T

def twist2axangle(T):
  '''
  converts an n x 4 x 4 twist (se3) matrix to an n x 6 axis-angle 
  '''
  return T[...,[0,1,2,2,0,1],[3,3,3,1,2,0]]

def axangle2adtwist(x):
  '''
  @Input:
    x = n x 6 = n elements of position and axis-angle
  @Output:
    A = n x 6 x 6 = n elements of ad(se(3))
  '''
  A = np.zeros(x.shape+(6,))
  A[...,0,1] =-x[...,5]
  A[...,0,2] = x[...,4]
  A[...,0,4] =-x[...,2]
  A[...,0,5] = x[...,1]
  
  A[...,1,0] = x[...,5]
  A[...,1,2] =-x[...,3]
  A[...,1,3] = x[...,2]
  A[...,1,5] =-x[...,0]
  
  A[...,2,0] =-x[...,4]
  A[...,2,1] = x[...,3]
  A[...,2,3] =-x[...,1]
  A[...,2,4] = x[...,0]
  
  A[...,3,4] =-x[...,5] 
  A[...,3,5] = x[...,4] 
  A[...,4,3] = x[...,5]
  A[...,4,5] =-x[...,3]   
  A[...,5,3] =-x[...,4]
  A[...,5,4] = x[...,3]
  return A

def twist2pose(T):
  '''
  converts an n x 4 x 4 twist (se3) matrix to an n x 4 x 4 pose (SE3) matrix 
  '''
  rotang = np.sqrt(np.sum(T[...,[2,0,1],[1,2,0]]**2,axis=-1)[...,None,None]) # n x 1
  Tn = np.nan_to_num(T / rotang)
  Tn2 = Tn@Tn
  Tn3 = Tn@Tn2
  eye = np.zeros_like(T)
  eye[...,[0,1,2,3],[0,1,2,3]] = 1.0
  return eye + T + (1.0 - np.cos(rotang))*Tn2 + (rotang - np.sin(rotang))*Tn3
  
def axangle2pose(x):
  '''
  @Input:
    x = n x 6 = n elements of position and axis-angle
  @Output:
    T = n x 4 x 4 = n elements of SE(3)
  '''
  return twist2pose(axangle2twist(x))


def pose2adpose(T):
  '''
  converts an n x 4 x 4 pose (SE3) matrix to an n x 6 x 6 adjoint pose (ad(SE3)) matrix 
  '''
  calT = np.empty(T.shape[:-2]+(6,6))
  calT[...,:3,:3] = T[...,:3,:3]
  calT[...,:3,3:] = axangle2skew(T[...,:3,3]) @ T[...,:3,:3]
  calT[...,3:,:3] = np.zeros(T.shape[:-2]+(3,3))
  calT[...,3:,3:] = T[...,:3,:3]
  return calT

def homegenous_transformation(R, t):
  T = np.eye(4)
  T[:3, :3] = R
  T[:3, 3] = t
  return T

def hat_map(x):   
  '''
  @Input:
    x = n x 3 = n elements 3d vectors
  @Output:
    x_hat = n x 3 x 3 = n elements of skew symmetric matrices
  '''
  x_hat = np.zeros((x.shape[0], 3, 3))
  x_hat[:, 0, 1] = -x[:, 2]
  x_hat[:, 0, 2] = x[:, 1]
  x_hat[:, 1, 0] = x[:, 2]
  x_hat[:, 1, 2] = -x[:, 0]
  x_hat[:, 2, 0] = -x[:, 1]
  x_hat[:, 2, 1] = x[:, 0]
  return x_hat

def generate_hat_map_matrices(w,v):
  twist_hat = np.zeros((w.shape[0], 4, 4))
  twist_hat[:, :3, :3] = w
  twist_hat[:, :3, 3] = v
  return twist_hat

def transform_pose_matrix_to_xy(pose_matrix,need_theata=False):
  init_pose = homegenous_transformation(np.eye(3),np.zeros(3))
  X = []
  Y = []
  Theta = []
  for i in range(len(pose_matrix)):
      position = pose_matrix[i] @ init_pose 
      x = position[:3, 3][0]
      y = position[:3, 3][1]
      X.append(x)
      Y.append(y)
      r = Rotation.from_matrix(position[:3, :3])
      deg = r.as_euler('zyx', degrees=False)
      Theta.append(deg[0])
  if need_theata == True:
      return X, Y, Theta
  else:
      return X, Y

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

def old_landmark_initialization(features,POSE,imu_T_cam,b):
    # relative transformation from left camera to right camera
    p = np.array([0,-b,0]).reshape(3,1)
    e_3 = np.array([0,0,1]).reshape(3,1)
    R = np.eye(3)
    o_T_r = np.array([[0,-1,0,0],
                [0,0,-1,0],
                [1,0, 0,0],
                [0,0, 0,1]])
    o_T_i = o_T_r @ np.linalg.inv(imu_T_cam)
    x_all = []
    y_all = []
    for j in range(1):
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
    return x_all,y_all

def landmark_initialization(features,POSE,imu_T_cam,K_s):
    # x_all = []
    # y_all = []
    # m_all = []
    observed = np.zeros(features.shape[1])
    landmark = np.ones((3, features.shape[1])) * -1
    for i in range(features.shape[2]):
        features_t = features[:,:,i]
        # keypoint observation: 1, 2, 3 ...
        index = np.where(np.min(features_t, axis=0) != -1)
        # previously unobserved landmark: 2, 3, ...
        index_unobserved = np.where(observed == 0)
        index_observed = np.where(observed == 1)
        unobserved = np.intersect1d(index, index_unobserved)
        obs = np.intersect1d(index, index_observed)
        features_t = features_t[:, unobserved]
        pts = features_t
        # calculate the disparity between left to the right: uL - uR = 1/z * fx b
        disparity = pts[0, :] - pts[2, :]
        fx_b = - K_s[2, -1]
        # calculate the depth estimation of the 3D points M(2,3)/uL-uR
        z = fx_b / disparity
        # since M is uninvertible, we need to solve the equation manually
        pts_3D = np.ones((4, pts.shape[1]))
        pts_3D[2, :] = z
        # x = uL * z / fx, y = uR * z / fy
        pts_3D[0, :] = (pts[0, :] * z - K_s[0,2] * z) / K_s[0,0]
        pts_3D[1, :] = (pts[1, :] * z - K_s[1,2] * z) / K_s[1,1]
        # calculate the [x, y, z, 1] in the optical frame
        X_r = imu_T_cam @ pts_3D
        m_w = POSE[i] @ X_r
        observed[unobserved] = 1
        landmark[:, unobserved] = m_w[:3, :]/m_w[-1, :]
        # m_all.append(m_w)
        # x_all.append(m_w[0,:])
        # y_all.append(m_w[1,:])
    return landmark

if __name__ == '__main__':
   print("This is a library of utility functions for pr3.")



