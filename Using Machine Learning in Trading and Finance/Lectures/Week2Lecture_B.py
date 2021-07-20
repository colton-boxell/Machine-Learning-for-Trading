import tensorflow as tf
from tensorflow import keras
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#Define your model
#A deeper neural network
model = tf.keras.models.Sequential([
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dense(10, activation='softmax')
	])

#Configure and train
model.compile(optimizer='adam', 
	loss='sparse_categorical_crossentropy', 
	metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)

#Once trained the model can be used for prediction
#Predictions returns a Numpy array of predictions
#With the predict() method you can pass A Dataset instance, 
# Numpy array, a Tensorflow tensor or list of tensors, or a generator of input samples
#Steps determines the total number of steps before declaring
# the prediction round finished. Here, since we have just one
# examples, steps = 1.
predictions = model.predict(input_samples, step=1)
print(predictions)