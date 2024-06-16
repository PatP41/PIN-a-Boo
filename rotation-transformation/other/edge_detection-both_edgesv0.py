# Trying to find the longest and kind of perpendicular edges of the phone to calculate the angles for the rotational matrix.
# A lot of errors.
# Lots of print statements because of debugging reasons.

import cv2
import numpy as np


def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges


def detect_lines(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
    return lines


def find_longest_line(lines):
    longest_line = max(lines, key=lambda line: np.sqrt((line[0][2] - line[0][0]) ** 2 + (line[0][3] - line[0][1]) ** 2),
                       default=None)
    return longest_line


def find_perpendicular_lines(longest_line, lines):
    x1, y1, x2, y2 = longest_line[0]
    angle_long = np.arctan2(y2 - y1, x2 - x1)
    angle_perpendicular = angle_long + np.pi / 2  # Perpendicular angle to find the shorter edge of the phone

    print(f"Angle of long edge: {np.degrees(angle_long):.2f} degrees")
    print(f"Perpendicular angle: {np.degrees(angle_perpendicular):.2f} degrees")

    perpendicular_lines = []
    for line in lines:
        if np.array_equal(line, longest_line):
            continue
        x3, y3, x4, y4 = line[0]
        angle = np.arctan2(y4 - y3, x4 - x3)
        angle_diff = abs(angle - angle_perpendicular)
        print(
            f"Detected line angle: {np.degrees(angle):.2f} degs, angle diff: {np.degrees(angle_diff):.2f} degs")
        if np.isclose(angle_diff, 0, atol=np.pi / 6):  # Allow for larger deviation
            perpendicular_lines.append(line)

    if perpendicular_lines:
        longest_perpendicular_line = max(perpendicular_lines, key=lambda line: np.sqrt(
            (line[0][2] - line[0][0]) ** 2 + (line[0][3] - line[0][1]) ** 2), default=None)
        return longest_perpendicular_line
    return None


def calculate_angle(line):
    x1, y1, x2, y2 = line[0]
    angle = np.arctan2(y2 - y1, x2 - x1)
    return np.degrees(angle)


def draw_angle(image, line, angle, color=(0, 0, 255)):
    x1, y1, x2, y2 = line[0]
    cv2.line(image, (x1, y1), (x2, y2), color, 5)
    mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2

    angle_text = f"{angle:.2f} degs"
    cv2.putText(image, angle_text, (mid_x + 10, mid_y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return image


def main():
    # Read and resize the image
    frame = cv2.imread("../assets/test/test5.jpg")
    frame = cv2.resize(frame, (500, 500))

    # Detect edges
    edges = detect_edges(frame)

    # Detect lines using Hough Transform
    lines = detect_lines(edges)

    if lines is not None:
        # Find the longest edge
        long_edge_line = find_longest_line(lines)

        if long_edge_line is not None:
            # Find the perpendicular short edge
            short_edge_line = find_perpendicular_lines(long_edge_line, lines)

            if short_edge_line is not None:
                # Calculate the angles of the long and short edge lines
                angle_long = calculate_angle(long_edge_line)
                angle_short = calculate_angle(short_edge_line)

                # Draw the angles on the original image
                frame = draw_angle(frame, long_edge_line, angle_long, color=(0, 255, 0))
                frame = draw_angle(frame, short_edge_line, angle_short, color=(255, 0, 0))
            else:
                print("Could not find a perpendicular line.")
                return
        else:
            print("Could not find the long edge.")
            return
    else:
        print("No lines detected :(")
        return

    print(f"Angle of the long edge: {angle_long:.2f} degs")
    print(f"Angle of the short edge: {angle_short:.2f} degs")

    # Show the images
    cv2.imshow("Both edges and angles", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
