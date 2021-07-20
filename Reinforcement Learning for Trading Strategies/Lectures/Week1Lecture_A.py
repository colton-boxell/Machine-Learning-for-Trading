import numpy as np

LAKE = np.array([[0, 0, 0, 0],
				[0, -1, 0, -1],
				[0, 0, 0, -1],
				[-1, 0, 0, 1]])
LAKE_WIDTH = len(LAKE[0])
LAKE_HEIGHT = len(LAKE)

DISCOUNT = .9 # Change me to be a value between 0 and 1.
DELTA = .0001 # I must be sufficiently small.
current_values = np.zeros_like(LAKE)
change = 1

def iterate_value(current_values):
	"""Finds the future state values for an array of current states.

	Args:
		current_values(int array): the value of current states.

	Returns:
		prime_values (int array): The value of states based on future states.
		policies (int array): The recommended action to take in a state.
	"""
	prime_values = []
	policies = []

	for state in STATE_RANGE:
		value, policy = get_max_neighbor(state, current_values)
		prime_values.append(value)
		policies.append(policy)

	prime_values = np.array(prime_values).reshape((LAKE_HEIGHT, LAKE_WIDTH))
	return prime_values, policies

# TODO: define get max neighbour function. Checks for neighboring black holes and rocks.

while change > DELTA:
	prime_values, policies = iterate_value(current_values)
	old_values = np.copy(current_values)
	current_values = DISCOUNT * prime_values
	change = np.sum(np.abs(old_values - current_values))

print(current_values)