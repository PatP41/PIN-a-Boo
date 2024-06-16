# Clusters the fingertip positions per frame and checks for proximity. If > threshold is in proximity, a keypress is detected
# Input a list of the normalised 2D! path coordinates

import cv2
import numpy as np


def define_number_block_dots():
    # Define the number block layout
    number_block_dots = {
        '1': (0.165, 0.125),
        '2': (0.495, 0.125),
        '3': (0.825, 0.125),
        '4': (0.165, 0.375),
        '5': (0.495, 0.375),
        '6': (0.825, 0.375),
        '7': (0.165, 0.625),
        '8': (0.495, 0.625),
        '9': (0.825, 0.625),
        '0': (0.495, 0.875)
    }
    return number_block_dots


def get_closest_number(x, y, number_block):
    closest_number = None
    min_dist = float('inf')
    for number, (cx, cy) in number_block.items():
        dist = np.linalg.norm(np.array([x, y]) - np.array([cx, cy]))
        if dist < min_dist:
            min_dist = dist
            closest_number = number
    return closest_number


def map_path_to_number_block_with_clusters(normalized_path, number_block, distance_threshold=0.05, cluster_threshold=3):
    pressed_numbers = []
    cluster = []

    for point in normalized_path:
        if not cluster:
            cluster.append(point)
        else:
            last_point = cluster[-1]
            dist = np.linalg.norm(np.array(point) - np.array(last_point))
            if dist < distance_threshold:
                cluster.append(point)
            else:
                if len(cluster) >= cluster_threshold:
                    avg_point = np.mean(cluster, axis=0)
                    number = get_closest_number(avg_point[0], avg_point[1], number_block)
                    if number and (not pressed_numbers or pressed_numbers[-1] != number):
                        pressed_numbers.append(number)
                cluster = [point]

    if len(cluster) >= cluster_threshold:
        avg_point = np.mean(cluster, axis=0)
        number = get_closest_number(avg_point[0], avg_point[1], number_block)
        if number and (not pressed_numbers or pressed_numbers[-1] != number):
            pressed_numbers.append(number)

    return pressed_numbers


def visualize_number_block_and_path(normalized_path, pressed_numbers, size=(360, 640)):
    img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    number_block_dots = define_number_block_dots()

    # Draw number block dots
    for _, (cx, cy) in number_block_dots.items():
        pt = (int(cx * size[0]), int(cy * size[1]))
        cv2.circle(img, pt, 5, (255, 255, 0), -1)

    # Draw the path
    for i, point in enumerate(normalized_path):
        x, y = point
        pt = (int(x * size[0]), int(y * size[1]))
        cv2.circle(img, pt, 5, (255, 0, 255), -1)
        if i > 0:
            prev_pt = (int(normalized_path[i - 1][0] * size[0]), int(normalized_path[i - 1][1] * size[1]))
            cv2.line(img, prev_pt, pt, (255, 0, 255), 2)

    # Display the guessed PIN
    pressed_text = 'Guessed PIN: ' + ' '.join(pressed_numbers)
    cv2.putText(img, pressed_text, (10, size[1] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imshow("Number Block with Path", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Test data
normalized_path = [
    # Cluster for '6'
    (0.8, 0.4), (0.76, 0.41), (0.75, 0.45), (0.76, 0.43), (0.79, 0.44),
    # Movement from '6' to '7'
    (0.6, 0.5), (0.4, 0.53), (0.22, 0.58),
    # Cluster for '7'
    (0.1, 0.6), (0.11, 0.65), (0.12, 0.63), (0.11, 0.64), (0.1, 0.62),
    # Movement from '7' to '2'
    (0.21, 0.5), (0.30, 0.41), (0.40, 0.28), (0.46, 0.16),
    # Cluster for '2'
    (0.5, 0.1), (0.51, 0.11), (0.52, 0.12), (0.51, 0.13), (0.5, 0.14),
    # Movement from '2' to '0'
    (0.50, 0.2), (0.51, 0.4), (0.49, 0.6), (0.50, 0.7), (0.52, 0.8),
    # Cluster for '0'
    (0.5, 0.85), (0.52, 0.86), (0.48, 0.87), (0.49, 0.88), (0.51, 0.89)
]

number_block = define_number_block_dots()
pressed_numbers = map_path_to_number_block_with_clusters(normalized_path, number_block)
visualize_number_block_and_path(normalized_path, pressed_numbers)
