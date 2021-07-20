def hurst(input_ts, lags_to_test=[2,20]):
	# interpretation of return value
	# hurst < 0.5 - input_tst is mean reverting
	# hurst = 0.5 - input_ts is effectively random/geometric brownian motion
	# hurst > 0.5 - input_ts is trending
	tau = []
	lagvec = []
	# Step through the different lags
	for lag in range(lags_to_test[0], lags_to_test[1]):
		# produce time series difference with lag
		pp = np.subtract(input_ts[lag:], input_ts[:-lag])
		# Write the different lags into a vector
		lagvec.append(lag)
		# Calculate the variance of the difference vector
		tau.append(np.std(pp))
	# linear fir to double-log graph (gives power)
	m = np.polyfit(np.log10(lagvec), np.log10(tau), 1)
	# hurst exponent is the slope of the line of best fit
	hurst = m[0]
	return hurst