import tensorflow as tf

def build_networks(state_shape, action_size, actor_lr, critic_lr, neurons):
		state_input = layers.Input(state_shape, name='frames')
		advantage = layers.Input((1,), name='A') # Now A instead of G.
		
		hidden_1 = layers.Dense(hidden_neurons, activation='relu')(state_input)
		hidden_2 = layers.Dense(hidden_neurons, activation='relu')(hidden_1)
		probabilities = layers.Dense(hidden_neurons, activation='softmax')(hidden_2)
		value = layers.Dense(1, activation='linear')(hidden_2)

		def custom_loss(y_true, y_pred):
			y_pred_clipped = K.clip(y_pred, 1e-8, 1-1e-8)
			log_likelihood = y_true * K.log(y_pred_clipped)
			return K.sum(-log_likelihood*g)

		actor = Model(inputs=[state_input, advantages], outputs=[probabilities, values])
		actor.compile(loss=[custom_loss, 'mean_squared_error'], optimizer=Adam(lr=actor_lr))

		critic = Model(inputs=[state_input], outputs=[value])
		predict = Model(inputs=[state_input], outputs=[probabilities])
		return actor, critic, predict

class Memory():
	def __init__(self, gamma):
		self.buffer = []
		self.gamma = gamma

	def add(self, experience)
		self.buffer.append(experience)

	def sample(self):
		batch = np.array(self.buffer).T.tolist()
		states_mb = np.array(batch[0], dtype=np.float32)
		actions_mb = np.array(batch[1]. dtype=np.int8)
		rewards_mb = np.array(batch[2], dtype=np.float32)
		self.buffer = []
		return states_mb, actions_mb, rewards_mb
	def learn(self):
		"""Trains the Deep Q Network based on stored experiences."""
		# Obtain a random mini-batch from memory.
		state_mb, action_mb, reward_mb = self.memory.sample()
		actions = tf.one_hot(action_mb, self.action_size)

		# Apply TD[0]
		discount_mb = reward_mb + next_v_mb * self.memory.gamma * (1 - dones_mb)
		state_values = self.critic.predict([state_mb])
		advantages = discount_mb - np.squeeze(state_values)
		self.actor.train_on_batch([state_mb, advantages], [action_mb, discount_mb])

	def act(self, state):
		state_batch = np.expand_dims(state, axis=0)
		probabilities = self.predict.predict(state_batch)[0]
		action = np.random.choice(self.action_size, p=probabilities)
		return action