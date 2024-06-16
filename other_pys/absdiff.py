# Using contour detection to show difference in frames
# First two as separate functions
# Below the combination of both


import cv2
import numpy as np


def contours():
    cap = cv2.VideoCapture(0)

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    while cap.isOpened():
        frame1 = cv2.resize(frame1, (500, 500))
        frame2 = cv2.resize(frame2, (500, 500))
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contoures, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame1, contoures, -1, (0, 255, 0), 2)
        cv2.imshow("Contours", frame1)
        frame1 = frame2
        ret, frame2 = cap.read()

        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()


def rect():
    cap = cv2.VideoCapture(0)
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    while cap.isOpened():
        frame1 = cv2.resize(frame1, (500, 500))
        frame2 = cv2.resize(frame2, (500, 500))
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            # Filter out small contours to reduce noise
            if cv2.contourArea(contour) < 15000:
                continue
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Rectangles", frame1)
        frame1 = frame2
        ret, frame2 = cap.read()
        if not ret:
            break
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# rect()
# contours()

def process_frame(frame1, frame2):
    # Resize for better processing
    frame1 = cv2.resize(frame1, (500, 500))
    frame2 = cv2.resize(frame2, (500, 500))

    # Computing absdiff
    diff = cv2.absdiff(frame1, frame2)

    # Convert to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply thresholding to get a binary image
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    # Dilate the binary image to fill in gaps
    dilated = cv2.dilate(thresh, None, iterations=3)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Copies needed to to both
    frame_contours = frame1.copy()
    frame_rects = frame1.copy()

    # Draw contours
    cv2.drawContours(frame_contours, contours, -1, (0, 255, 0), 2)

    # Draw rectangles
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 15000:
            continue
        cv2.rectangle(frame_rects, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame_contours, frame_rects


def add_title(image, title):
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(image, title, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def combine_images(img1, img2, title1, title2):
    img1 = add_title(img1, title1)
    img2 = add_title(img2, title2)

    combined_image = np.hstack((img1, img2))
    return combined_image


def main():
    cap = cv2.VideoCapture(0)

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    while cap.isOpened():
        frame_contours, frame_rects = process_frame(frame1, frame2)

        combined_image = combine_images(frame_contours, frame_rects, "Contours", "Rectangles")

        cv2.imshow("Contours and Rectangles", combined_image)

        frame1 = frame2
        ret, frame2 = cap.read()

        if not ret:
            break

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
