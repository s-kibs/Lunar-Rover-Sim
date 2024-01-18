import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

# from tensorflow.keras.losses import SparseCategoricalCrossentropy
# model.compile(loss=SparsCategoricalCrossentropy)

predictions = model(x_train[:1]).numpy()
predictions
tf.nn.softmax(predictions).numpy()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test,  y_test, verbose=2)
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
probability_model(x_test[:5])

#test example a varying variable
test_example = 6785 
input_test_image = x_test[test_example]
true_label = y_test[test_example]


input_test_image = np.expand_dims(input_test_image, axis=0)
predictions_to_test = model.predict(input_test_image)
predicted_test_label = np.argmax(predictions_to_test)


plt.imshow(x_test[test_example], cmap='gray')
plt.title(f'True Label: {true_label}, Predicted Label: {predicted_test_label}')
plt.show()
