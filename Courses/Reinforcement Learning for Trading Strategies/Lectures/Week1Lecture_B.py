def iteratire_policy(current_values, current_policies)
	""" Finds the future states values for an array of current states.

		Args:
			current_values (int array): the value of current states.
			current_policies (int array): a list where each cell is the recommended
				action for the state matching its index.

		Returns:
			next_values (int array): The value of states based on future states.
			next_policies (int array): The recommended action to take in a state.
	"""
	next_values = find_future_values(current_values, current_policies)
	next_policies = find_best_policy(next_values)
	return next_values, next_policies

def find_future_values(current_values, current_policies):
	"""Finds the next set of future values basedon the current policy."""
	next_values = []

	for state in STATE_RANGE:
		current_policy = current_policies[state]
		state_x, state_y = get_state_coordinates(state)

		# If the cell has something other than 0, it's a terminal state.
		value = LAKE[state_y, state_x]
		if not value:
			value = get_neighbor_value(
				state_x, state_y, current_values, current_policy)
			next_values.append(value)
		return np.array(next_values).reshape((LAKE_HEIGHT, LAKE_WIDTH))

def find_best_policy(next_values):
	"""Finds the best policy given a value mapping."""
	next_policies = []
	for state in STATE_RANGE:
		state_x, state_y = get_state_coordinates(state)

		# No policy or best value yet
		max_value = -np.inf
		best_policy = -1

		if not LAKE[state_y, state_x]:
			for policy in ACTION_RANGE:
				neighbor_value = get_neighbor_value(
					state_x, state_y, next_values, policy)
				if neighbor_value > max_value:
					max_value = neighbor_value
					best_policy = policy
		next_policies.append(best_policy)
	return next_policies