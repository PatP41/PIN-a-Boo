# Line detector using Hough grid
# Doesn't work well in this instance -> scrapped

import cv2
import numpy as np


def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    rho = 1  # distance resolution of the Hough grid in pixels
    theta = np.pi / 180  # angular resolution of the Hough grid in radians
    threshold = 15  # minimum number of intersections in Hough grid cell (votes)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(frame) * 0

    # Run Hough on edge detected image
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        line = line.ravel()
        print(line)
        # print(cv2.arcLength(line,closed=True)) # Not working currently -> ignore
    lines_edges = cv2.addWeighted(frame, 0.8, line_image, 1, 0)

    return lines_edges


def main():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        lines_edges = process_frame(frame)

        cv2.imshow("Hough", lines_edges)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
