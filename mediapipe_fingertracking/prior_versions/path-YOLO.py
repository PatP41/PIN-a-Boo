# Show path when finger (in this instance: thumb) goes into the bounding box.
# Added a grace period to account for errors
# No zoom into ROI yet

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


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No camera")
        exit()

    hand_tracker = HandTracker()
    yolo_model = YOLO('../../rotation-transformation/yolov8n.pt')
    draw_list = []
    show_pattern = False
    grace_period = 10  # Number of frames to keep tracking the thumb when no phone found
    frames_without_box = 0

    while True:
        success, image = cap.read()
        if not success:
            print("Can't retrieve frame :(")
            break

        # YOLO Detection
        results = yolo_model.track(source=image, classes=67, conf=0.5, tracker="bytetrack.yaml", persist=True)
        annotated_frame = results[0].plot() if results else image

        boxes = results[0].boxes.cpu().numpy() if results else []
        thumb_inside_box = False

        # Hand Tracking
        image = hand_tracker.find_hands(image)
        lm_list = hand_tracker.find_position(image)

        # Check if thumb tip is inside bbox
        if lm_list:
            thumb_tip = lm_list[4]
            thumb_x, thumb_y, thumb_z = thumb_tip[1], thumb_tip[2], thumb_tip[3]
            for box in boxes:
                xa, ya, xb, yb = box.xyxy[0].astype(int)
                cv2.rectangle(image, (xa, ya), (xb, yb), (0, 255, 0), 2)
                if xa <= thumb_x <= xb and ya <= thumb_y <= yb:
                    thumb_inside_box = True
                    frames_without_box = 0
                    break

        if thumb_inside_box:
            draw_list.append([thumb_x, thumb_y, thumb_z])
            show_pattern = True
        else:
            frames_without_box += 1

        # Continue tracking for a few frames even if no phone detected / needed because this yolo is not perfect
        if frames_without_box < grace_period and lm_list:
            draw_list.append([thumb_x, thumb_y, thumb_z])

        if show_pattern:
            pattern = np.zeros_like(image)
            for i, point in enumerate(draw_list):
                cv2.circle(pattern, (point[0], point[1]), 5, (255, 0, 255), cv2.FILLED)
                if i > 0:
                    cv2.line(pattern, (draw_list[i - 1][0], draw_list[i - 1][1]), (point[0], point[1]), (255, 0, 255),
                             2)
            cv2.imshow("Thumb Path", pattern)

        if lm_list:
            thumb_tip = lm_list[4]
            cv2.circle(image, (thumb_tip[1], thumb_tip[2]), 10, (0, 0, 255), cv2.FILLED)

        cv2.imshow("Video", image)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
