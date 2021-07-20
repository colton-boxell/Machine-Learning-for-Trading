"""From my earlier failures, I knew that no matter how confident I was in making
any one bet I could still be wrong - and that proper diversification was the key
to reducing risks without reducing returns. If I could build a portfolio filled
with high-quality return streams that were properly diversified (they zigged and
zagged in ways that balanced each other out), I could offer clients an overall
portfolio return much more consistent and reliable than what they coudl get
elsewhere. -Life and Work by Ray Dalio """

#%config InlineBackend.figure_format = "retina"

import numpy as np
import pandas as pd
import altair as alt

np.random.seed(42) # Set seed for reproducibility

def correlated_streams(n, mean, risk, corr):
	"""Generates 'n' return streams with given average 'mean' and
	'risk'
	and with an average correlation level 'corr'.
	"""

	num_samples = 10000
	means = np.full(n, mean)

	corr_mat = np.full((n, n), corr, dtype=np.dtype("d"))
	np.fill_diagonal(corr_mat, 1,)
	cov_mat = corr_mat * risk**2

	streams = np.random.multivariate_normal(means, cov_mat, size=num_samples)

	return streams.T

#Sanity check
n = 5
mean, std, corr = 10, 15, 6
streams = correlated_streams(n, mean, std, corr)
print(streams.mean(axis=1))
print(streams.std(axis=1))
np.corrcoef(streams)

def aggregate_risk(return_streams, n):
	"""Returns the pooled risk (std) of the 'n' first streams
	in 'return_streams'
	"""
	assert len(return_streams) >= n

	aggregate_returns = np.sum(return_streams[:n], axis=0)/n
	return aggregate_returns.std()

#Building dataset
max_assets = 20
assets = range(1, max_assets+1)

mean = 10 # Avg mean return of 10%
risk_levels = range(1,15)

index = pd.MultiIndex.from_product([risk_levels, assets],
	names = ["risk_level", "num_assets"])
simulated_data = pd.DataFrame(index=index)

for risk in risk_levels:
	for corr in np.arange(0.0, .8, 0.1):
		return_streams = correlated_streams(max_assets, mean, risk, corr)
		risk_level = np.zeros(max_assets)
		for num_assets in assets:
			risk_level[num_assets-1] = aggregate_risk(return_streams, num_assets)
		simulated_data.loc[(risk,),round(corr,1)] = risk_level
	simulated_data.columns.names = ["correlation"]

simulated_data.query("risk_level == 14")

def plot_risk_level(data, risk_level):
	subset = data.query("risk_level == @risk_level")
	stacked = subset.stack().reset_index(name="risk")
	stacked.head()

	chart = alt.Chart(data=stacked)

	highlight = alt.selection(type='single', on="mouseover",
		field=["correlation"], nearest=True)

	base = chart.encode(
		alt.X("num_assets", axis=alt.Axis(title="Number of Assets")),
		alt.Y("risk", axis=alt.Axis(title="Risk %")),
		alt.Color("correlation.N", scale=alt.Scale(scheme="set2")))

	points = base.mark_circle().encode(
		opacity=alt.value(0)).add_selection(highlight).properties(
		height=400, 
		width=600, 
		title="Risk % by number of asset in portfolio"
		)
	lines = base.mark_line().encode(
		size=alt.condition(~highlight, alt.value(1), alt.value(3)),
		tooltip=["correlation"])
	return points + lines

plot_risk_level(simulated_data, 10)