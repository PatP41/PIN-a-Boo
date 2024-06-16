# Simple transform/rotation
# can be used to illustrate the angle calculations and rotations

import cv2
import numpy as np

# Test data: finger path coords (x, y)
finger_path = [
[0.51, 0.18],
[0.5, 0.22],
[0.54, 0.25],
[0.52, 0.23],
[0.54, 0.21],
[0.53, 0.41],
[0.49, 0.6],
[0.48, 0.79],
[0.46, 0.91],
[0.51, 0.92],
[0.49, 0.9],
[0.5, 0.91],
[0.48, 0.92],
[0.4, 0.77],
[0.35, 0.66],
[0.26, 0.52],
[0.17, 0.42],
[0.12, 0.36],
[0.14, 0.36],
[0.15, 0.35],
[0.16, 0.36],
[0.16, 0.38],
[0.22, 0.4],
[0.41, 0.46],
[0.59, 0.54],
[0.69, 0.57],
[0.79, 0.58],
[0.83, 0.62],
[0.85, 0.6],
[0.84, 0.65],
[0.85, 0.64],
[0.87, 0.62]
]

finger_path = np.array(finger_path)

# Test angle
angle = 70

angle_rad = np.radians(angle)

# Rotation matrix for rotation
R = np.array([
    [np.cos(angle_rad), -np.sin(angle_rad)],
    [np.sin(angle_rad), np.cos(angle_rad)]
])

# Apply the rotation matrix to each point in the finger path
transformed_finger_path = np.dot(finger_path - 0.5, R.T) + 0.5

image_size = (640, 360, 3)
original_image = np.ones(image_size, dtype=np.uint8) * 255
transformed_image = np.ones(image_size, dtype=np.uint8) * 255

def draw_path(image, path, color=(255, 0, 0)):
    for i in range(len(path) - 1):
        start_point = (int(path[i][0] * image_size[1]), int(path[i][1] * image_size[0]))
        end_point = (int(path[i + 1][0] * image_size[1]), int(path[i + 1][1] * image_size[0]))
        cv2.line(image, start_point, end_point, color, 2)
    for point in path:
        cv2.circle(image, (int(point[0] * image_size[1]), int(point[1] * image_size[0])), 5, color, -1)

draw_path(original_image, finger_path, (255, 0, 0))
draw_path(transformed_image, transformed_finger_path, (0, 0, 255))
print(transformed_finger_path)

combined_image = np.hstack((original_image, transformed_image))

cv2.imshow("Finger path: Orginal and Transformation", combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
