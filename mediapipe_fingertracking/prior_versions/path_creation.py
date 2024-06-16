# Press s to start showing the path of the thumb.
# Simple path tracking
# YOLO is not implemented yet.

import cv2
import mediapipe as mp
import numpy as np


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
                lm_list.append([id, cx, cy, lm.x, lm.y, lm.z])
        return lm_list


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    hand_tracker = HandTracker()
    draw_list = []
    show_pattern = False

    while True:
        success, image = cap.read()
        if not success:
            print("Can't retrieve frame...")
            break

        # Hand Tracking
        image = hand_tracker.find_hands(image)
        lm_list = hand_tracker.find_position(image)

        # Draw hand landmarks
        if hand_tracker.results.multi_hand_landmarks:
            for hand_landmarks in hand_tracker.results.multi_hand_landmarks:
                hand_tracker.mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        # Display the video feed with hand landmarks
        cv2.imshow("Video", image)

        # Use 's' to show key press
        if cv2.waitKey(1) == ord('s'):
            show_pattern = True

        # Track thumb tip and draw path
        if show_pattern and lm_list:
            try:
                draw_list.append(lm_list[4][1:3])
            except IndexError:
                pass

            pattern = np.zeros_like(image)
            for i, point in enumerate(draw_list):
                cv2.circle(pattern, (point[0], point[1]), 5, (255, 0, 255), cv2.FILLED)
                if i > 0:
                    cv2.line(pattern, tuple(draw_list[i - 1]), tuple(draw_list[i]), (255, 0, 255), 2)

            cv2.imshow("Thumb Path", pattern)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
