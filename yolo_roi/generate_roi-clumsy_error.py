# Has an error message that pops up as soon as the bounding box is not detected.
# Use generate_roi-better_error-padding.py or generate_roi-better_error_no-padding.py instead


import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO('../../yolov8n.pt')


def crop_image(image, bbox):
    if bbox is None:
        return image
    x1, y1, x2, y2 = map(int, bbox)
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

    x_offset = (target_width - new_size[0]) // 2
    y_offset = (target_height - new_size[1]) // 2

    padded_roi[y_offset:y_offset + new_size[1], x_offset:x_offset + new_size[0]] = scaled_roi

    return padded_roi


# Start video capture
cap = cv2.VideoCapture(0)

while True:
    suc, frame = cap.read()
    if not suc:
        break

    results = model.track(source=frame, show=False, classes=67, tracker="bytetrack.yaml", persist=True)

    roi_image = create_placeholder(frame.shape)

    for result in results:
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            bboxes = boxes.xyxy.tolist()  # Each box is in (x1, y1, x2, y2) format

            bbox = bboxes[0]

            roi_image = crop_image(frame, bbox[:4])

            roi_image = scale_and_pad_roi(roi_image, frame.shape)

            x1, y1, x2, y2 = map(int, bbox[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            break

    composite_image = np.hstack((frame, roi_image))

    cv2.imshow("Composite Frame", composite_image)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
