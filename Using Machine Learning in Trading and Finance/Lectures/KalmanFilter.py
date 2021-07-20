from pykalman import KalmanFilter
import numpy as np


kf = KalmanFilter(n_dims=1, d_dim_state=2,
				initial_state_mean=np.zeros(2),
				initial_state_covariance=np.ones((2,2)),
				transition_matrices=np.eye(2),
				observation_matrices=obs_mat,
				observation_covariance=1.0,
				transition_covariance=trans_conv)