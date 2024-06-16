# Calculating a rotational matrix using the angle from the longest edge

import cv2
import numpy as np


def find_longest_line(lines):
    max_len = 0
    longest_line = None
    for line in lines:
        for x1, y1, x2, y2 in line:
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length > max_len:
                max_len = length
                longest_line = (x1, y1, x2, y2)
    return longest_line


def calculate_angle(line):
    x1, y1, x2, y2 = line
    angle = np.arctan2(y2 - y1, x2 - x1)
    return np.degrees(angle)


def compute_rotation_matrix(angle):
    angle_rad = np.radians(angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rotation_matrix = np.array([  # standard 2d rotation-transformation matrix, not 3d sadly
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    return rotation_matrix


def draw_angle(image, line, angle):
    x1, y1, x2, y2 = line
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
    mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2

    radius = 50
    angle_start = angle
    angle_end = angle + 180 if angle < 0 else angle - 180

    arc_start = (int(mid_x + radius * np.cos(np.radians(angle_start))),
                 int(mid_y - radius * np.sin(np.radians(angle_start))))
    arc_end = (int(mid_x + radius * np.cos(np.radians(angle_end))),
               int(mid_y - radius * np.sin(np.radians(angle_end))))

    # cv2.ellipse(image, (mid_x, mid_y), (radius, radius), 0, angle_start, angle_end, (0, 0, 255), 2)
    cv2.line(image, (mid_x, mid_y), (mid_x, mid_y + 50), (0, 0, 255), 2)
    cv2.line(image, (mid_x, mid_y), (mid_x + 50, mid_y), (0, 0, 255), 2)

    return image


def main():
    test = cv2.imread("../assets/test/test2.jpg")
    test = cv2.resize(test, (1000, 1000))
    img = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    scale_percent = 60
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    test = cv2.resize(test, dim, interpolation=cv2.INTER_AREA)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # Detect edges
    edges = cv2.Canny(img, 50, 120)

    # Detect lines using Hough
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180.0, threshold=20, minLineLength=20, maxLineGap=5)

    # Find the longest line
    longest_line = find_longest_line(lines)

    if longest_line is not None:
        x1, y1, x2, y2 = longest_line
        cv2.line(test, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Calculate the angle of the longest line
        angle = calculate_angle(longest_line)
        print(f"Angle of phone edge: {angle:.2f} deg")
        print(f"Angle2 of phone edge: {90 + angle:.2f} deg")

        # Compute rotation-transformation matrix
        rotation_matrix = compute_rotation_matrix(angle)
        rotation_matrix2 = compute_rotation_matrix(90 + angle)  # different sources calculate different angles? shrug
        print("Rotation Matrix:")
        print(rotation_matrix)
        print("Rotation Matrix2:")  # same stuff but entries a bit scrambled
        print(rotation_matrix2)

        test = draw_angle(test, longest_line, angle)

    # cv2.imshow("edges", edges)
    cv2.imshow("lines", test)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
