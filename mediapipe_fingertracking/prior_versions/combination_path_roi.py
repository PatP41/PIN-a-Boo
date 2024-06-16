# Combination_v1 of path-yolo.py and generate_roi-better_error-padding.py
# Has the problem of showing "Thumb Error" if the bounding box is gone

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO


class HandTracker:
    def __init__(self, mode=False, max_hands=1, detection_confidence=0.5, model_complexity=1, tracking_confidence=0.5):
        self.hands = mp.solutions.hands.Hands(
            mode, max_hands, model_complexity, detection_confidence, tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def find_hands(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image_rgb)
        return image

    def find_position(self, image, hand_no=0):
        lm_list = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(hand.landmark):
                h, w, _ = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cz = lm.z
                lm_list.append([id, cx, cy, cz, lm.x, lm.y, lm.z])
        return lm_list


def crop_image(image, bbox):
    if bbox is None:
        return image  # Return the original image if no bbox
    x1, y1, x2, y2 = map(int, bbox)  # Convert coords to ints / important!
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


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No camera")
        exit()

    hand_tracker = HandTracker()
    yolo_model = YOLO('../../rotation-transformation/yolov8n.pt')
    draw_list = []
    show_pattern = False
    grace_period = 20
    frames_without_box = 0

    last_roi_image = create_placeholder((480, 640, 3), "No Phone Detected")
    last_bbox = None

    while True:
        success, frame = cap.read()
        if not success:
            print("Can't retrieve frame :(")
            break

        # YOLO Detection
        results = yolo_model.track(source=frame, show=False, classes=67, conf=0.5, tracker="bytetrack.yaml",
                                   persist=True)
        boxes = results[0].boxes.cpu().numpy() if results else []
        thumb_inside_box = False

        # Hand Tracking
        frame = hand_tracker.find_hands(frame)
        lm_list = hand_tracker.find_position(frame)

        # Check if thumb tip is inside any bbox
        if lm_list:
            thumb_tip = lm_list[4]
            thumb_x, thumb_y, thumb_z = thumb_tip[1], thumb_tip[2], thumb_tip[3]
            for box in boxes:
                xa, ya, xb, yb = box.xyxy[0].astype(int)
                if xa <= thumb_x <= xb and ya <= thumb_y <= yb:
                    thumb_inside_box = True
                    frames_without_box = 0
                    last_bbox = [xa, ya, xb, yb]
                    break

        if thumb_inside_box:
            draw_list.append([thumb_x, thumb_y, thumb_z])
            show_pattern = True
        else:
            frames_without_box += 1

        # Initialize the ROI image to the last detected ROI or placeholder
        if frames_without_box < grace_period and last_bbox is not None:
            roi_image = crop_image(frame, last_bbox)
            roi_image = scale_and_pad_roi(roi_image, frame.shape)
            last_roi_image = roi_image
        elif frames_without_box >= grace_period:
            last_roi_image = create_placeholder(frame.shape, "No Phone Detected")
            draw_list = []
            show_pattern = False

        # Create a composite image
        if thumb_inside_box and show_pattern:
            pattern = np.zeros_like(last_roi_image)
            for i, point in enumerate(draw_list):
                cv2.circle(pattern, (point[0], point[1]), 5, (255, 0, 255), cv2.FILLED)
                if i > 0:
                    cv2.line(pattern, (draw_list[i - 1][0], draw_list[i - 1][1]), (point[0], point[1]), (255, 0, 255),
                             2)
            composite_image = np.hstack((last_roi_image, pattern))
        elif last_bbox is not None and frames_without_box < grace_period:
            composite_image = np.hstack((last_roi_image, create_placeholder(frame.shape, "No Thumb Detected")))
        else:
            composite_image = np.hstack((create_placeholder(frame.shape, "No Phone Detected"),
                                         create_placeholder(frame.shape, "No Phone Detected")))

        cv2.imshow("Composite Frame", composite_image)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
