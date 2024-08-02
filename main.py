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

from dataclasses import dataclass
import cv2 as cv
import numpy as np
import math

cap = cv.VideoCapture("video/fuen2.mov")

win1 = "frame"
win2 = "bg"
cv.namedWindow(win1)
cv.namedWindow(win2)
cv.moveWindow(win1, 0, 0)
cv.moveWindow(win2, 0, 390)


"""
1. Preprocessing
- Background modeling using MOG over 300 frames
"""
backSub = cv.createBackgroundSubtractorKNN(
    history=300, dist2Threshold=400, detectShadows=False
)


@dataclass
class PossibleWave:
    contour: np.ndarray
    rect: np.ndarray
    box: np.ndarray
    area: float
    inertia_ratio: float

    def __init__(self, contour, rect, box, area, inertia_ratio):
        self.contour = contour
        self.rect = rect
        self.box = box
        self.area = area
        self.inertia_ratio = inertia_ratio

    def draw(self, frame):
        # Draw the contour
        cv.drawContours(frame, [self.contour], -1, (0, 255, 0), 1)

        # Ensure text positions are within the frame bounds
        x, y = int(self.box[0][0]), int(self.box[0][1])
        text_area = f"area: {self.area:.2f}"
        text_inertia = f"inertia_ratio: {self.inertia_ratio:.2f}"

        # Define font properties
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 255, 0)
        thickness = 1

        # Get text size for bounding box
        text_size, _ = cv.getTextSize(text_area, font, font_scale, thickness)

        # Draw the area text
        cv.putText(frame, text_area, (x, y), font, font_scale, color, thickness)

        # Draw the inertia ratio text
        cv.putText(
            frame,
            text_inertia,
            (x, y + text_size[1] + 5),
            font,
            font_scale,
            color,
            thickness,
        )

    def __repr__(self):
        return f"PossibleWave(area={self.area}, inertia_ratio={self.inertia_ratio})"

    def __str__(self):
        return f"PossibleWave(area={self.area}, inertia_ratio={self.inertia_ratio})"

    def __lt__(self, other):
        return self.area < other.area

    def __eq__(self, other):
        return self.area == other.area

    def __gt__(self, other):
        return self.area > other.area

    def __le__(self, other):
        return self.area <= other.area

    def __ge__(self, other):
        return self.area >= other.area

    def __ne__(self, other):
        return self.area != other.area


def calculate_inertia_ratio(moments):
    # Calculate the denominator using the correct normalization
    denominator = math.sqrt(
        (2 * moments["m11"]) ** 2 + (moments["m20"] - moments["m02"]) ** 2
    )

    # Small epsilon to avoid division by zero
    epsilon = 0.01
    if denominator < epsilon:
        return 0.0  # handle division by zero or near-zero

    # Calculate the sin and cos of the angle
    cosmin = (moments["m20"] - moments["m02"]) / denominator
    sinmin = 2 * moments["m11"] / denominator
    cosmax = -cosmin
    sinmax = -sinmin

    # Calculate the minimum and maximum inertia
    imin = (
        0.5 * (moments["m20"] + moments["m02"])
        - 0.5 * (moments["m20"] - moments["m02"]) * cosmin
        - moments["m11"] * sinmin
    )

    imax = (
        0.5 * (moments["m20"] + moments["m02"])
        - 0.5 * (moments["m20"] - moments["m02"]) * cosmax
        - moments["m11"] * sinmax
    )

    ratio = imin / imax
    return ratio


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
    if len(contours) > 0:
        filtered_contours = [
            c
            for c in contours[0]
            if cv.contourArea(c) > 200
            and cv.contourArea(c) < 1000
            and calculate_inertia_ratio(cv.moments(c)) < 0.01
        ]

        for c in filtered_contours:
            wave = PossibleWave(
                c,
                cv.minAreaRect(c),
                cv.boxPoints(cv.minAreaRect(c)),
                cv.contourArea(c),
                calculate_inertia_ratio(cv.moments(c)),
            )
            wave.draw(frame)
            # rect = cv.minAreaRect(c)
            # box = cv.boxPoints(rect)
            # box = box.astype(int)
            # cv.drawContours(frame, [c], -1, (0, 255, 0), 1)

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
