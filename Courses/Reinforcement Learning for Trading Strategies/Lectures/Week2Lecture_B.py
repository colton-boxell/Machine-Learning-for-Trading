import tensorflow as tf

def build_networks(state_shape, action_size, learning_rate, hidden_neurons):
		state_input = Input(state_shape, name='frames')
		g = Input((1,), name='G')
		hidden_1 = Dense(hidden_neurons, activation='relu')(state_input)
		hidden_2 = Dense(hidden_neurons, activation='relu')(hidden_1)
		probabilities = Dense(hidden_neurons, activation='softmax')(hidden_2)

		def custom_loss(y_true, y_pred):
			y_pred_clipped = K.clip(y_pred, 1e-8, 1-1e-8)
			log_likelihood = y_true * K.log(y_pred_clipped)
			return K.sum(-log_likelihood*g)

		policy = Model(
			inputs=[state_input, g], outputs=[probabilities])
		optimizer = Adam(lr=learning_rate)
		policy.compile(loss=custom_loss, optimizer=optimizer)

		predict = Model(inputs=[state_input], outputs=[probabilities])
		return policy, predict

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

		# Normalized TD(1)
		discount_mb = np.zeros_like(reward_mb)
		total_rewards = 0
		for t in reversed(range(len(reward_mb))):
			total_rewards = reward_mb[t] + total_rewards * self.memory.gamma
			discount_mb[t] = total_rewards
		discount_mb = (discount_mb - np.mean(discount_mb)) / np.std(discount_mb)

		self.policy.train_on_batch([state_mb, discount_mb], actions)

	def act(self, state):
		state_batch = np.expand_dims(state, axis=0)
		probabilities = self.predict.predict(state_batch)[0]
		action = np.random.choice(self.action_size, p=probabilities)
		return action