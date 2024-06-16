# Comparing LSD, edges and Hough for consistency.

import cv2
import numpy as np


def process_frame(frame):
    # Resize frame for better processing
    frame = cv2.resize(frame, (1000, 1000))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    scale_percent = 60
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    dim = (width, height)

    frame_resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    gray_resized = cv2.resize(gray, dim, interpolation=cv2.INTER_AREA)

    edges = cv2.Canny(gray_resized, 50, 120)

    lines = cv2.HoughLinesP(edges, rho=1,
                            theta=np.pi / 180.0,
                            threshold=20,
                            minLineLength=20,
                            maxLineGap=5)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame_resized, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # Create default parametrization LSD
    lsd = cv2.createLineSegmentDetector(0)

    # Detect lines with LSD
    lines_lsd = lsd.detect(edges)[0]  # pos 0 = detected lines

    drawn_img = lsd.drawSegments(gray_resized, lines_lsd)

    return frame_resized, edges, drawn_img


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No camera")
        exit()

    while True:
        success, frame = cap.read()
        if not success:
            print("Can't retrieve frame :(")
            break

        frame_processed, edges, drawn_img = process_frame(frame)

        cv2.imshow("Original Frame with Hough Lines", frame_processed)
        cv2.imshow("Edges", edges)
        cv2.imshow("LSD Lines", drawn_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
