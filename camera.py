import cv2
import numpy as np


def estimate_camera_position(image, grid_size, fixed_distance=254):
    """
    Estimates the camera position using a calibration pattern image and a fixed distance.

    Args:
        image (np.ndarray): The image to analyze.
        grid_size (int): The number of squares in the calibration grid.
        fixed_distance (float, optional): The fixed distance between the camera and the object in millimeters. Defaults to 254.

    Returns:
        np.ndarray: The estimated camera position in millimeters, or None if estimation fails.
    """

    # Load the calibration pattern image (replace with your actual path)
    pattern_image = cv2.imread("calibration_pattern.png")

    # Convert images to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_pattern_image = cv2.cvtColor(pattern_image, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners in both images
    ret1, corners1 = cv2.findChessboardCorners(gray_pattern_image, (grid_size, grid_size), None)
    ret2, corners2 = cv2.findChessboardCorners(gray_image, (grid_size, grid_size), None)

    # Check if corners are found in both images
    if not ret1 or not ret2:
        print("Error: Chessboard corners not found!")
        return None

    # Calculate the object size based on the grid size and known square size (adjust accordingly)
    object_size = (grid_size - 1) * 10  # Assuming each square is 10mm wide

    # Calculate the focal length using the formula
    # f = object_size * pixel_size / distance
    # where pixel_size is the average distance between two corresponding corners in pixels
    pixel_size = np.mean(np.linalg.norm(corners2 - corners1, axis=1))
    focal_length = object_size * pixel_size / fixed_distance

    # Calculate the rotation vector and translation vector (simplified approach)
    retval, rvecs, tvecs = cv2.solvePnP(corners1, corners2, np.array([[focal_length, 0, image.shape[1] / 2],
                                                                 [0, focal_length, image.shape[0] / 2],
                                                                 [0, 0, 1]]), None, None, None, cv2.SOLVE_PNP_ITERATIVE)

    # Extract the camera position (inverted for easier visualization)
    camera_position = -tvecs[0]

    return camera_position


# Example usage
def capture_and_estimate_position(grid_size, fixed_distance=254):
    """
    Captures an image from the default camera and estimates the camera position.

    Args:
        grid_size (int): The number of squares in the calibration grid.
        fixed_distance (float, optional): The fixed distance between the camera and the object in millimeters. Defaults to 254.

    Returns:
        np.ndarray: The estimated camera position in millimeters, or None if capture or estimation fails.
    """

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Failed to open camera!")
        return None

    ret, frame = cap.read()

    cap.release()

    if not ret:
        print("Error: Failed to capture frame!")
        return None

    camera_position = estimate_camera_position(frame, grid_size, fixed_distance)

    return camera_position


if __name__ == "__main__":
    grid_size = 7  # Number of squares in the calibration grid (adjust accordingly)

    camera_position = capture_and_estimate_position(grid_size)

    if camera_position is not None:
        print("Estimated camera position:", camera_position)
    else:
        print("Failed to estimate camera position")
