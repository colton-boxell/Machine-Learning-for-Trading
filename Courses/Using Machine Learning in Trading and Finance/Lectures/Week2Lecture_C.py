INPUT_COLS = [
	'pickup_longitude',
	'pickup_latitude',
	'dropoff_longitude',
	'dropoff_latitude',
	'passenger_count'
	]

# Prepare input feature columns
inputs = {colname: layers.Input(name=colname, shape=(), dtype='float32')
		for colname in INPUT_COLS
}

# Create deep columns
deep_columns = [
	# Embedding_column to "group" together ...
	fc.embedding_column(fc_crossed_pd_pair, 10),

	# Numeric columns
	fc.numeric_column("pickup_latitude"),
	fc.numeric_column("pickup_longitude"),
	fc.numeric_column("dropoff_longitude"),
	fc.numeric_column("dropoff_latitude")]

# Create the deep part of the model
deep_inputs = layers.DenseFeatures
	(deep_columns, name='deep_inputs')(inputs)
x = layers.Dense(30, activation='relu')(deep_inputs)
x = layers.Dense(20, activation='relu')(x)

deep_output = layers.Dense(10, activation='relu')(x)

# Create wide columns
wide_columns = [
	# One-hot encoded feature crosses
	fc.indicator_column(fc_crossed_dloc),
	fc.indicator_column(fc_crossed_ploc),
	fc.indicator_column(fc_crossed_pd_pair)
]

# Create the wide part of the model
wide = layers.DenseFeatures(wide_columns, name='wide_inputs')(inputs)

# Combine outputs
combined = concatenate(inputs=[deep,wide],
						name='combined')
output = layers.Dense(1,
					activation=None,
					name='prediction')(combined)

# Finalize model
model = keras.Model(inputs=list(inputs.values()),
					name='wide_and_deep')
model.compile(optimizer='adam',
			loss='mse',
			metrics=[rmse, "mse"])