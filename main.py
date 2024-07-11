#
"""
Reimplementation of https://github.com/citrusvanilla/multiplewavetracking_py as a learning experiment

Near-shore ocean wave recognition through Computer Vision "recognition" workflow 
for video sequences

Pipeline consists of 4 steps

1. Preprocessing
2. Object detection
3. Object tracking
4. Object recognition/classification

"""

import cv2 as cv

cap = cv.VideoCapture("video/fuengirola.mp4")

ret, frame = cap.read()
cv.imshow("frame", frame)

cv.waitKey(0)
cap.release()
cv.destroyAllWindows()
