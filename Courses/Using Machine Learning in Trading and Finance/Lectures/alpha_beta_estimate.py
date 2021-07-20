import numpy as np
import pandas_datareader as pdr
from pykalman import KalmanFilter
import matplotlib.pyplot as plt

# Load pricing data for a security
start = '2017-11-01'
end = '2019-11-01'
AAPL = pdr.DataReader('AAPL',data_source="yahoo", start=start, end=end)
AAPL = AAPL['Adj Close']
SPY = pdr.DataReader('LMT',data_source="yahoo", start=start, end=end)
SPY = SPY['Adj Close']
x = AAPL

delta = 1e-3
trans_conv = delta/(1-delta)*np.eye(2) # How much random walk wiggles
obs_mat = np.expand_dims(np.vstack([[x],np.ones(len(x))]).T, axis=1)

kf = KalmanFilter(n_dim_obs=1, n_dim_state=2, 
				# y is 1-dimensional, (alpha, beta) is 2-dimensional
				initial_state_mean=[0,0],
				initial_state_covariance=np.ones((2,2)),
				transition_matrices=np.eye(2),
				observation_matrices=obs_mat,
				observation_covariance=2,
				transition_covariance=trans_conv)
# Use the observations y to get running estimates and errors for the state parameters
state_means,state_covs = kf.filter(x.values)
_,axarr = plt.subplots(2,sharex=True)
axarr[0].plot(x.index,state_means[:,0],label='slope')
axarr[0].legend()
axarr[1].plot(x.index, state_means[:,1], label='intercept')
axarr[1].legend()
plt.tight_layout()
plt.show()