import cv2
import numpy as np
import config


class MotionDetector:

    def __init__(self):
        self.back_sub = cv2.createBackgroundSubtractorMOG2(
            history=config.MOG2_HISTORY,
            varThreshold=config.MOG2_VAR_THRESHOLD,
            detectShadows=False
        )
        self.kernel = np.ones(config.KERNEL_SIZE, np.uint8)
        self.prev_frame = None

    def _get_mog2_mask(self, blurred):
        mask = self.back_sub.apply(blurred)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.dilate(mask, self.kernel, iterations=config.DILATE_ITERATIONS)
        return mask

    def _get_frame_diff_mask(self, blurred):
        if self.prev_frame is None:
            self.prev_frame = blurred
            return None
        diff = cv2.absdiff(self.prev_frame, blurred)
        _, diff_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        diff_mask = cv2.dilate(diff_mask, self.kernel, iterations=2)
        self.prev_frame = blurred
        return diff_mask

    def _get_confidence(self, mog2_mask, diff_mask):
        if diff_mask is None:
            return "LOW"
        overlap = cv2.bitwise_and(mog2_mask, diff_mask)
        overlap_area = cv2.countNonZero(overlap)
        mog2_area = cv2.countNonZero(mog2_mask)
        diff_area = cv2.countNonZero(diff_mask)
        if overlap_area > config.MIN_CONTOUR_AREA and mog2_area > 0 and diff_area > 0:
            return "HIGH"
        elif mog2_area > config.MIN_CONTOUR_AREA or diff_area > config.MIN_CONTOUR_AREA:
            return "LOW"
        else:
            return "NONE"

    def _merge_boxes(self, boxes):
        if not boxes:
            return []
        x_min = min(x for x, y, w, h in boxes)
        y_min = min(y for x, y, w, h in boxes)
        x_max = max(x + w for x, y, w, h in boxes)
        y_max = max(y + h for x, y, w, h in boxes)
        return [(x_min, y_min, x_max - x_min, y_max - y_min)]

    def process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, config.BLUR_KERNEL, 0)
        mog2_mask = self._get_mog2_mask(blurred)
        diff_mask = self._get_frame_diff_mask(blurred)
        confidence = self._get_confidence(mog2_mask, diff_mask)
        contours, _ = cv2.findContours(mog2_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) < config.MIN_CONTOUR_AREA:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))
        boxes = self._merge_boxes(boxes)
        return boxes, confidence