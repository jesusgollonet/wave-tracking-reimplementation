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
# cap = cv.VideoCapture("video/original/scene2.mp4")

win1 = "frame"
win2 = "bg"
cv.namedWindow(win1)
cv.namedWindow(win2)
cv.moveWindow(win2, 0, 390)


"""
1. Preprocessing
- Background modeling using MOG over 300 frames
"""
backSub = cv.createBackgroundSubtractorMOG2(
    history=300, varThreshold=100, detectShadows=False
)

while 1:
    ret, frame = cap.read()
    if not ret:
        break
    # bg subtraction
    fgMask = backSub.apply(frame)
    # remove noise by opening

    opening = cv.morphologyEx(
        fgMask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    )
    # show frame number
    cv.putText(
        fgMask,
        str(int(cap.get(cv.CAP_PROP_POS_FRAMES))),
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2,
    )
    cv.imshow(win1, frame)
    cv.imshow(win2, opening)
    if cv.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
