import numpy as np

class Memory():
	def __init__(self, memory_size, batch_size):
		self.buffer = deque(maxlen=memory_size)
		self.batch_size = batch_size

	def add(self, experience):
		# Adds a (state, action, reward, state_prime, done) tuple.
		self.buffer.append(experience)

	def sample(self):
		buffer_size = len(self.buffer)
		index = np.random.choice(
			np.arrange(buffer_size), size=self.batch_size, replace=False)
		batch = [self.buffer[i] for i index]
		return batch