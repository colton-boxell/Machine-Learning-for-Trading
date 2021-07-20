import numpy as np
import pandas as pd
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
y = SPY

# Construct a Kalman filter
kf = KalmanFilter(transition_matrices=[1],
				observation_matrices=[1],
				initial_state_mean = 0,
				initial_state_covariance=1,
				observation_covariance=1,
				transition_covariance=.01)

# User the observed values of the price to get a rolling mean
state_means,_ = kf.filter(x.values)
state_means = pd.Series(state_means.flatten(), index=x.index)

# Compute the rolling mean with various lookback windows
mean50 = x.rolling(window=50).mean()
mean100 = x.rolling(window=100).mean()

# Plot original data and estimated mean
plt.plot(state_means)
plt.plot(x)
plt.plot(mean50)
plt.plot(mean100)
plt.title('Kalman filter estimate of average')
plt.legend(['Kalman Estimate', 'X', '50-day Moving Average', '100-day Moving Average'])
plt.xlabel('Day')
plt.ylabel('Price')
plt.show()

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
state_means,state_covs = kf.filter(y.values)
_,axarr = plt.subplots(2,sharex=True)
axarr[0].plot(x.index,state_means[:,0],label='slope')
axarr[0].legend()
axarr[1].plot(x.index, state_means[:,1], label='intercept')
axarr[1].legend()
plt.tight_layout()
plt.show()

# Plot data point using colormap
sc = plt.scatter(x,y,s=30,edgecolor='k',alpha=0.7)
cb = plt.colorbar(sc)
# TODO set ytick locations
#cb.ax.set_yticks()
#cb.ax.set_yticklabels([str(p.date()) for p in x[::len(x)//9].index])

# Plot every 5th line
step = 5
xi = np.linspace(x.min()-5,x.max()+5,2)
colors_l=np.linspace(0.1,1,len(state_means[::step]))
for i, beta in enumerate(state_means[::step]):
	plt.plot(xi,beta[0]*xi + beta[1], alpha=.2,lw=1)

# Plot the OLS regression line
plt.plot(xi,np.poly1d(np.polyfit(x,y,1))(xi), '0.4')

# Adjust axes for visibility
plt.axis([225,300,125,250])

# Label axes
plt.xlabel('SPY')
plt.ylabel('APPL')
plt.show()

# Get returns from pricing data
x_r=x.pct_change()[1:]
y_r=y.pct_change()[1:]

# Run Kalman filter on returns data
delta_r = 1e-2
trans_cov_r = delta_r/(1 - delta_r)*np.eye(2) 
# How much random walk wiggles
obs_mat_r = np.expand_dims(np.vstack([[x_r],[np.ones(len(x_r))]]).T, axis=1)
kf_r = KalmanFilter(n_dim_obs=1, n_dim_state=2, 
				# y_r is 1-dimensional, (alpha, beta) is 2-dimensional
				initial_state_mean=[0,0],
				initial_state_covariance=np.ones((2,2)),
				transition_matrices=np.eye(2),
				observation_matrices=obs_mat,
				observation_covariance=.01,
				transition_covariance=trans_cov_r)

_,axarr = plt.subplots(2,sharex=True)
axarr[0].plot(x.index,state_means[:,0],label='slope')
axarr[0].legend()
axarr[1].plot(x.index,state_means[:,1],label='intercept')
axarr[1].legend()
plt.tight_layout()
plt.show()