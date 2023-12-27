import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generate random data
np.random.seed(0)
X = np.random.rand(100, 1)
Y = 2 * X + 1 + 0.1 * np.random.rand(100, 1)

# Define the model
model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])

# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train the model
model.fit(X, Y, epochs=100)

# Make predictions
predictions = model.predict(X)

# Plot the results
plt.scatter(X, Y)
plt.plot(X, predictions, color='red')
plt.show()
