from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_cnn_model(input_shape):
  """
  Creates a simple CNN model for lettuce height estimation.

  Args:
      input_shape: A tuple representing the input image shape (height, width, channels).

  Returns:
      A compiled CNN model for lettuce height estimation.
  """

  model = Sequential([
      Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
      MaxPooling2D((2, 2)),
      Conv2D(64, (3, 3), activation='relu'),
      MaxPooling2D((2, 2)),
      Flatten(),
      Dense(128, activation='relu'),
      Dense(1)  # Output layer for height prediction (regression)
  ])
  model.compile(loss='mse', optimizer='adam')
  return model


# (Optional) Load preprocessed training and validation data (using load_data from data_processing.py)
train_images, train_heights = load_data(train_data_dir)
val_images, val_heights = load_data(val_data_dir)

# Train the model (replace with your training loop and hyperparameter tuning)
model = create_cnn_model(train_images.shape[1:])
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=20, shear_range=0.2, zoom_range=0.2)
val_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow(train_images, train_heights, batch_size=32)
val_generator = val_datagen.flow(val_images, val_heights, batch_size=32)
model.fit(train_generator, epochs=20, validation_data=val_generator)

# Save the model for future use (optional)
model.save('lettuce_height_estimator.h5')
