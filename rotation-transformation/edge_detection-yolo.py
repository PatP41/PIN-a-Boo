# Combination of the edge detection and the generate_roi...pys.
# Decent results with some errors that need manual debugging.

import cv2
import numpy as np
from ultralytics import YOLO


yolo_model = YOLO('yolov8n.pt')

def crop_image(image, bbox):
    if bbox is None:
        return image  # Return the original image if no bbox
    x1, y1, x2, y2 = map(int, bbox)  # Convert coords to ints
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image

def create_placeholder(image_shape, text="No Phone Detected"):
    placeholder = np.zeros(image_shape, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_COMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = (image_shape[1] - text_size[0]) // 2
    text_y = (image_shape[0] + text_size[1]) // 2
    cv2.putText(placeholder, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return placeholder

def scale_roi(roi, target_shape):
    target_height, target_width = target_shape[:2]
    roi_height, roi_width = roi.shape[:2]

    scale_ratio = min(target_width / roi_width, target_height / roi_height)
    new_size = (int(roi_width * scale_ratio), int(roi_height * scale_ratio))
    scaled_roi = cv2.resize(roi, new_size)

    return scaled_roi

def detect_edges(image):
    if image is None or image.size == 0:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def detect_lines(edges):
    if edges is None or edges.size == 0:
        return None
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
    return lines

def find_longest_line(lines):
    if lines is None or len(lines) == 0:
        return None
    longest_line = max(lines, key=lambda line: np.sqrt((line[0][2] - line[0][0]) ** 2 + (line[0][3] - line[0][1]) ** 2),
                       default=None)
    return longest_line

def find_perpendicular_lines(longest_line, lines):
    if longest_line is None or lines is None or len(lines) == 0:
        return None
    x1, y1, x2, y2 = longest_line[0]
    angle_long = np.arctan2(y2 - y1, x2 - x1)
    angle_perpendicular = angle_long + np.pi / 2  # Perpendicular angle

    perpendicular_lines = []
    for line in lines:
        if np.array_equal(line, longest_line):
            continue
        x3, y3, x4, y4 = line[0]
        angle = np.arctan2(y4 - y3, x4 - x3)
        angle_diff = abs(angle - angle_perpendicular)
        if np.isclose(angle_diff, 0, atol=np.pi / 6):  # Allow for larger deviation from perpendicular
            perpendicular_lines.append(line)

    if perpendicular_lines:
        longest_perpendicular_line = max(perpendicular_lines, key=lambda line: np.sqrt(
            (line[0][2] - line[0][0]) ** 2 + (line[0][3] - line[0][1]) ** 2), default=None)
        return longest_perpendicular_line
    return None

def calculate_angle(line):
    if line is None:
        return None
    x1, y1, x2, y2 = line[0]
    angle = np.arctan2(y2 - y1, x2 - x1)
    return np.degrees(angle)

def draw_angle(image, line, angle, color=(0, 0, 255)):
    if line is None or angle is None:
        return image
    x1, y1, x2, y2 = line[0]
    cv2.line(image, (x1, y1), (x2, y2), color, 5)
    mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2

    angle_text = f"{angle:.2f} degrees"
    cv2.putText(image, angle_text, (mid_x + 10, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return image

def main():
    cap = cv2.VideoCapture(0)

    # Initialize the last detected ROI and bounding box
    last_roi_image = None
    last_bbox = None

    no_detection_frames = 0
    max_no_detection_frames = 10  # Number of grace period frames

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        success, frame = cap.read()
        if not success:
            print("Can't retrieve frame...")
            break

        results = yolo_model.track(source=frame, show=False, classes=67, tracker="bytetrack.yaml", persist=True)

        # Initialize the ROI image to the last detected ROI or placeholder
        if last_roi_image is None or no_detection_frames >= max_no_detection_frames:
            roi_image = create_placeholder(frame.shape)
        else:
            if last_bbox is not None:
                roi_image = crop_image(frame, last_bbox)
                roi_image = scale_roi(roi_image, frame.shape)
            else:
                roi_image = last_roi_image

        detected = False
        for result in results:
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                bboxes = boxes.xyxy.tolist()  # Each box is in (x1, y1, x2, y2) format
                bbox = bboxes[0]
                roi_image = crop_image(frame, bbox[:4])
                roi_image = scale_roi(roi_image, frame.shape)
                last_roi_image = roi_image
                last_bbox = bbox[:4]
                no_detection_frames = 0
                detected = True
                break

        if not detected:
            no_detection_frames += 1

        if no_detection_frames < max_no_detection_frames:
            edges = detect_edges(last_roi_image)
            lines = detect_lines(edges)

            if lines is not None:
                long_edge_line = find_longest_line(lines)

                if long_edge_line is not None:
                    short_edge_line = find_perpendicular_lines(long_edge_line, lines)

                    if short_edge_line is not None:
                        angle_long = calculate_angle(long_edge_line)
                        angle_short = calculate_angle(short_edge_line)
                        print(f"Long edge angle: {angle_long:.2f} degs")
                        print(f"Short edge angle: {angle_short:.2f} degs")
                        last_roi_image = draw_angle(last_roi_image, long_edge_line, angle_long, color=(0, 255, 0))
                        last_roi_image = draw_angle(last_roi_image, short_edge_line, angle_short, color=(255, 0, 0))
                    else:
                        print("Could not find a perpendicular line to the longest edge.")
                else:
                    print("Could not find the longest edge.")
            else:
                print("No lines detected.")
        else:
            last_roi_image = create_placeholder(frame.shape, "No Phone Detected")

        if last_roi_image is None:
            last_roi_image = create_placeholder(frame.shape, "No Phone Detected")

        roi_image_resized = scale_roi(last_roi_image, frame.shape)
        roi_height, roi_width = roi_image_resized.shape[:2]
        composite_image = np.ones((frame.shape[0], frame.shape[1] + roi_width, 3), dtype=np.uint8) * 255
        composite_image[:, :frame.shape[1]] = frame
        composite_image[(frame.shape[0] - roi_height) // 2:(frame.shape[0] - roi_height) // 2 + roi_height,
        frame.shape[1]:] = roi_image_resized

        cv2.imshow("Composite Frame", composite_image)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
