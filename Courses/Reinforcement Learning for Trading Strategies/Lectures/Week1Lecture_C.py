# Anatomy of an Agent
import numpy as np
import gym

class Agent():
	def __init__(num_states, num_actions, discount, learning_rate, random_rate):
		self.discount = discount
		self.learning_rate = learning_rate
		self.num_actions = num_actions
		self.random_rate = random_rate # Between 0 and 1
		self.q_table = np.zeros((num_states, num_actions))

	def update_q(self, state, action, reward, state_prime)
		alpha = self.learning_rate
		future_value = reward + self.discount * np.max(q_table[state_prime])
		old_value = q_table[state, action]
		q_table[state, action] = old_value + alpha * (future_value - old_value)

	def act(self, state):
		if random.random() < self.random_rate and training:
			return random.randint(0, self.num_actions - 1)
		action_values = q_table[state_row]
		max_indexes = np.argwhere(action_values == action_values.max())
		max_indexes = np.squeeze(max_indexes, axis = -1)
		action = np.random.choice(max_indexes)
		return action

EPISODES = 1000
agent = Agent(NUM_STATES, NUM_ACTIONS, DISCOUNT, LEARNING_RATE, RANDOM_RATE)
environment = gym.make('FrozenLake-v0')

def play_game(environment, agent):
	state = environment.reset()
	done = False

	while not done:
		action = agent.act(state)
		state_prime, reward, done = environment.step(action)
		agent.update_q(state, action, reward, state_prime)
		state = new_state

for episode in range(EPISODES):
	play_game(environment, agent)	