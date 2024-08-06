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

from wavetracker.utils import calculate_inertia_ratio
import cv2 as cv


cap = cv.VideoCapture("video/fuen2.mov")

win1 = "frame"
win2 = "bg"
cv.namedWindow(win1)
cv.namedWindow(win2)
cv.moveWindow(win1, 0, 0)
cv.moveWindow(win2, 0, 390)


"""
1. Preprocessing
- Background modeling using KNN over 300 frames
"""
backSub = cv.createBackgroundSubtractorKNN(
    history=300, dist2Threshold=400, detectShadows=False
)


tracked_waves = []

while 1:
    ret, frame = cap.read()

    if not ret:
        break

    # bg subtraction
    fgMask = backSub.apply(frame)
    # remove noise by opening
    kernel_size = 5
    structuring_element = cv.getStructuringElement(
        cv.MORPH_RECT, (kernel_size, kernel_size)
    )
    opening = cv.morphologyEx(fgMask, cv.MORPH_OPEN, structuring_element, iterations=1)
    """
    2.Detection 
    find the contours, filter them by area and draw boudning box
    """
    contours = cv.findContours(opening, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # filter contours by area and inertia ratio (elongation)
    cols, rows = frame.shape[:2]
    filtered_contours = [
        c
        for c in contours[0]
        if cv.contourArea(c) > 40
        and cv.contourArea(c) < 2000
        and calculate_inertia_ratio(cv.moments(c)) < 0.0001
    ]
    for c in filtered_contours:
        rect = cv.minAreaRect(c)
        box = cv.boxPoints(rect)
        area = cv.contourArea(c)
        inertia_ratio = calculate_inertia_ratio(cv.moments(c))
        moments = cv.moments(c)
        x = int(moments["m10"] / moments["m00"])
        y = int(moments["m01"] / moments["m00"])
        for tw in tracked_waves:
            if cv.pointPolygonTest(tw.contour, (x, y), False) == 1:
                tw.area = area
                tw.inertia_ratio = inertia_ratio
                tw.box = box
                tw.rect = rect
                tw.contour = c
                break
            tw.area = area
            tw.inertia_ratio = inertia_ratio
            tw.box = box
            tw.rect = rect
            tw.contour = c
            break

    cv.imshow(win1, frame)
    cv.imshow(win2, opening)
    if cv.waitKey(30) & 0xFF == ord("q"):
        break


cap.release()
cv.destroyAllWindows()
