# Error comes up after a few frames of no detection (10).
# Otherwise keeps the feed going where the last bounding box was detected.
# With black padding to keep the same size as main window.

import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO('yolov8n.pt')


def crop_image(image, bbox):
    if bbox is None:
        return image  # Return the original image if no bbox
    x1, y1, x2, y2 = map(int, bbox)  # Convert coords into ints
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image


def create_placeholder(image_shape):
    placeholder = np.zeros(image_shape, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_COMPLEX
    text = "No Phone Detected"
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = (image_shape[1] - text_size[0]) // 2
    text_y = (image_shape[0] + text_size[1]) // 2
    cv2.putText(placeholder, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return placeholder


def scale_and_pad_roi(roi, target_shape):
    target_height, target_width = target_shape[:2]
    roi_height, roi_width = roi.shape[:2]

    scale_ratio = min(target_width / roi_width, target_height / roi_height)
    new_size = (int(roi_width * scale_ratio), int(roi_height * scale_ratio))
    scaled_roi = cv2.resize(roi, new_size)

    padded_roi = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Calculations for padding
    x_offset = (target_width - new_size[0]) // 2
    y_offset = (target_height - new_size[1]) // 2

    # Scaled ROI with padding
    padded_roi[y_offset:y_offset + new_size[1], x_offset:x_offset + new_size[0]] = scaled_roi

    return padded_roi


cap = cv2.VideoCapture(0)

# Initialize the last detected ROI and bounding box
last_roi_image = None
last_bbox = None

# Frame counter for no detection
no_detection_frames = 0
max_no_detection_frames = 5  # Number grace period frames for the placeholder

# Track with the model
while True:
    suc, frame = cap.read()
    if not suc:
        break

    results = model.track(source=frame, show=False, classes=67, tracker="bytetrack.yaml", persist=True)

    # Initialize the ROI image to the last detected ROI or placeholder
    if last_roi_image is None or no_detection_frames >= max_no_detection_frames:
        roi_image = create_placeholder(frame.shape)
    else:
        if last_bbox is not None:
            roi_image = crop_image(frame, last_bbox)
            roi_image = scale_and_pad_roi(roi_image, frame.shape)
        else:
            roi_image = last_roi_image

    detected = False
    for result in results:
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            bboxes = boxes.xyxy.tolist()  # Each box is in (x1, y1, x2, y2) format
            # Use the first bounding box
            bbox = bboxes[0]
            # Crop the image
            roi_image = crop_image(frame, bbox[:4])
            # Scale and pad the ROI
            roi_image = scale_and_pad_roi(roi_image, frame.shape)
            last_roi_image = roi_image
            last_bbox = bbox[:4]
            no_detection_frames = 0
            detected = True
            x1, y1, x2, y2 = map(int, bbox[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            break

    if not detected:
        no_detection_frames += 1

    composite_image = np.hstack((frame, roi_image))

    cv2.imshow("Composite Frame", composite_image)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
