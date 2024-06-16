# Chang of line_edges-video.py to test one frame instead.

import cv2
import numpy as np

# Read gray image
test = cv2.imread("../assets/test/test2.jpg")
test = cv2.resize(test, (1000, 1000))
img = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
scale_percent = 60
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
test = cv2.resize(test, dim, interpolation=cv2.INTER_AREA)
img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
edges = cv2.Canny(img, 50, 120)

lines = cv2.HoughLinesP(edges, rho=1,
                        theta=np.pi / 180.0,
                        threshold=20,
                        minLineLength=20,
                        maxLineGap=5)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(test, (x1, y1), (x2, y2), (0, 255, 255), 5)

cv2.imshow("edges", edges)
cv2.imshow("lines", test)

lsd = cv2.createLineSegmentDetector(0)

leno = []
lines = lsd.detect(edges)[0]

# Finding the biggest line. Doesn't work well here
# for id, line in enumerate(lines):
# length = np.linalg.norm(abs(line.x1-line.x2),abs(line.y1-line.y2))
# leno.append(id, length)
# biggest = sort(leno)[-1:-2]
# print(biggest)
# Draw detected lines in the image

drawn_img = lsd.drawSegments(img, lines)

cv2.imshow("LSD", drawn_img)
cv2.waitKey(0)
