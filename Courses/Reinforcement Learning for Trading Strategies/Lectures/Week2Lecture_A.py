import numpy as np
import tensorflow as tf

class Memory():
	def __init__(self, memory_size, batch_size, random_rate, random_decay):
		self.buffer = deque(maxlen=memory_size)
		self.batch_size = batch_size
		self.random_rate = random_rate
		self.random_decay = random_decay

	def add(self, experience):
		# Adds a (state, action, reward, state_prime, done) tuple.
		self.buffer.append(experience)

	def sample(self):
		buffer_size = len(self.buffer)
		index = np.random.choice(
			np.arrange(buffer_size), size=self.batch_size, replace=False)
		batch = [self.buffer[i] for i index]
		return batch

	def deep_q_network(state_shape, action_size, learning_rate, hidden_neurons):
		state_input = Input(state_shape, name='frames')
		actions_input = Input((action_size,),name='mask')

		hidden_1 = Dense(hidden_neurons, activation='relu')(state_input)
		hidden_2 = Dense(hidden_neurons, activation='relu')(hidden_1)
		q_values = Dense(action_size)(hidden_2)
		masked_q_values = Multiply()([q_values, actions_input])

		model = Model(inputs=[state_input, action_input], outputs=masked_q_values)
		optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate)
		model.compile(loss='mse', optimizer=optimizer)
		return model

	def act(self, state, training=False):
		if training:
			# Random actions until enough simulations to train the model
			if len(self.memory.buffer) >= self.memory.batch_size:
				self.random_rate *= self.random_decay
			if self.random_rate >= np.random.rand():
				return random.randint(0, self.action_size-1)

		# If not acting randomly, take action with highest predicted value.
		state_batch = np.expand_dims(state, axis=0)
		predict_mask = np.ones((1, self.action_size,))
		actions_qs = self.network.predict([state_batch, predict_mask])
		return np.argmax(actions_qs[0])

	def update_Q(self):
		state_mb, action_mb, reward_mb, state_prime_mb, done_mb = (
			self.memory.sample())

		# Get Q values for state_prime_mb.
		predict_mask = np.ones(action_mb.shape + (self.action_size,))
		next_q_mb = self.network.predict([state_prime_mb, predict_mask])
		next_q_mb = tf.math.reduce_max(next_q_mb, axis=1)

		# Apply the Bellamn Equation
		target_qs = (next_q_mb * self.memory.gamma) + reward_mb
		target_qs = tf.where(done_mb, reward_mb, target_qs)

		# Match training batch to network output
		action_mb = tf.convert_to_tensor(action_mb, dtype=tf.int32)
		action_hot = tf.one_hot(action_mb, self.action_size)
		target_mask = tf.multiply(tf.expand_dims(target_qs, -1), action_hot)
		return self.network.train_on_batch([state_mb, action_hot], target_mask)