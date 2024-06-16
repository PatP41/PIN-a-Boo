# Clusters the fingertip positions per frame and checks for proximity. If > threshold is in proximity, a keypress is detected
# Input a list of the normalised 2D! path coordinates
# Visually different output

import cv2
import numpy as np


def define_number_block():
    # Define the number block layout
    number_block = {
        '1': (0.0, 0.0, 0.33, 0.25),
        '2': (0.33, 0.0, 0.66, 0.25),
        '3': (0.66, 0.0, 1.0, 0.25),
        '4': (0.0, 0.25, 0.33, 0.5),
        '5': (0.33, 0.25, 0.66, 0.5),
        '6': (0.66, 0.25, 1.0, 0.5),
        '7': (0.0, 0.5, 0.33, 0.75),
        '8': (0.33, 0.5, 0.66, 0.75),
        '9': (0.66, 0.5, 1.0, 0.75),
        '0': (0.33, 0.75, 0.66, 1.0)
    }
    return number_block


def map_path_to_number_block_with_clusters(normalized_path, number_block, distance_threshold=0.05, cluster_threshold=3):
    pressed_numbers = []
    cluster = []

    def get_number_from_point(x, y):
        for number, (x1, y1, x2, y2) in number_block.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                return number
        return None

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
                    number = get_number_from_point(avg_point[0], avg_point[1])
                    if number and (not pressed_numbers or pressed_numbers[-1] != number):
                        pressed_numbers.append(number)
                cluster = [point]

    # Check the last cluster
    if len(cluster) >= cluster_threshold:
        avg_point = np.mean(cluster, axis=0)
        number = get_number_from_point(avg_point[0], avg_point[1])
        if number and (not pressed_numbers or pressed_numbers[-1] != number):
            pressed_numbers.append(number)

    return pressed_numbers


def visualize_number_block_and_path(normalized_path, pressed_numbers, size=(300, 400)):
    img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    number_block = define_number_block()

    # Draw the layout
    for number, (x1, y1, x2, y2) in number_block.items():
        pt1 = (int(x1 * size[0]), int(y1 * size[1]))
        pt2 = (int(x2 * size[0]), int(y2 * size[1]))
        cv2.rectangle(img, pt1, pt2, (0, 0, 0), 2)
        text_pos = (pt1[0] + 10, pt1[1] + 30)
        cv2.putText(img, number, text_pos, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

    # Draw the path between fingertip per frame
    for i, point in enumerate(normalized_path):
        x, y = point
        pt = (int(x * size[0]), int(y * size[1]))
        cv2.circle(img, pt, 5, (0, 0, 255), -1)
        if i > 0:
            prev_pt = (int(normalized_path[i - 1][0] * size[0]), int(normalized_path[i - 1][1] * size[1]))
            cv2.line(img, prev_pt, pt, (0, 0, 255), 2)

    # Display the guessed PIN
    pressed_text = 'Guessed PIN: ' + ' '.join(pressed_numbers)
    cv2.putText(img, pressed_text, (10, size[1] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imshow("PIN on Number Block", img)
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
number_block = define_number_block()
pressed_numbers = map_path_to_number_block_with_clusters(normalized_path, number_block)
visualize_number_block_and_path(normalized_path, pressed_numbers)
