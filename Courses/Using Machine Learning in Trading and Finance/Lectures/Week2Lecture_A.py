import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard

((x_train, y_train),(x_test, y_test)) = keras.datasets.mnist.load_data()

#The keras model stacks layers on the top of each other
#The batch size is omitted. Here the model expects batch of vectors with 64 components
model =Sequentia([
	Input(shape=(64,))
	Dense(units=32. activation="relu", name="hidden1"),
	Dense(units=32. activation="relu", name="hidden2"),
	Dense(units=32. activation="linear", name="output")
	])

#Define your model
#A linear model( a single Dense layers) aka multiclass logisitic regression
model = tf.keras.models.Sequentia([
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(10, activation='softmax')
	])

#Define your model
#A neural network with one hidden layer
model = tf.keras.models.Sequentia([
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dense(10, activation='softmax')
	])

#Define your model
#A neural network with multiple hidden layers (a deep neural network)
model = tf.keras.models.Sequentia([
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dense(10, activation='softmax')
	])

#Define your model
#A deeper neural network
model = tf.keras.models.Sequentia([
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dense(10, activation='softmax')
	])

#compiling a Keras model

def rmse(y_true, y_pred):
	return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))
model.compile(optimizer="adam", loss="mse", metrics=[rmse,"mse"])

#Training a Keras model
steps_per_epoch = NUM_TRAIN_EXAMPLES // (TRAIN_BATCH_SIZE * NUM_EVALS)
#This is a trick so that we have control of the total number of examples the model trains
# on. (NUM_TRAIN_EXAMPLES) and the total number of evaluation we want to have during the
# training (NUM_EVALS).
histroy = model.fit(
	x=trainsds,
	steps_per_epoch = steps_per_epoch,
	epochs = NUM_EVALS,
	validation_data=evalds,
	callbacks=[TensorBoard(LOGDIR)]
	)

