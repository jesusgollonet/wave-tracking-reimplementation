import cv2 as cv


class Preprocessor:
    def __init__(self):
        self.backSub = cv.createBackgroundSubtractorKNN(
            history=300, dist2Threshold=400, detectShadows=False
        )

    def update(self, frame):
        fgMask = self.backSub.apply(frame)
        # remove noise by opening
        kernel_size = 5
        structuring_element = cv.getStructuringElement(
            cv.MORPH_RECT, (kernel_size, kernel_size)
        )
        opening = cv.morphologyEx(
            fgMask, cv.MORPH_OPEN, structuring_element, iterations=1
        )
        return opening
