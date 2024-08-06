import cv2 as cv
from wavetracker.utils import calculate_inertia_ratio


class Detector:
    def __init__(self):
        print("Detector initialized")

    def update(self, frame):
        contours = cv.findContours(frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # filter contours by area and inertia ratio (elongation)
        cols, rows = frame.shape[:2]
        filtered_contours = [
            c
            for c in contours[0]
            if cv.contourArea(c) > 40
            and cv.contourArea(c) < 2000
            and calculate_inertia_ratio(cv.moments(c)) < 0.0001
        ]
        return filtered_contours
