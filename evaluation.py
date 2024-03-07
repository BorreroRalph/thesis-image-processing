from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error

# Path to your trained model (modify as needed)
model_path = 'lettuce_height_estimator.h5'

# Path to your test data (modify as needed)
test_data_dir = 'C:/Users/asus/Documents/test_data'  # Replace with actual path

def load_data(data_dir):
  """
  Loads images and actual heights from a directory (similar to data_processing.py).
  """
  # ... (implement similar logic to load_data from data_processing.py)
  return images, heights

def main():
  # Load the trained model
  model = load_model(model_path)

  # Load test data
  test_images, test_heights = load_data(test_data_dir)

  # Make predictions on test data
  predicted_heights = model.predict(test_images)

  # Calculate mean squared error (MSE)
  mse = mean_squared_error(test_heights, predicted_heights)
  print(f"Mean Squared Error (MSE): {mse:.2f} cm^2")

  # You can explore other evaluation metrics here (optional)
  # ... (e.g., root mean squared error, correlation coefficient)

if __name__ == '__main__':
  main()
