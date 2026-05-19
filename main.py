import cv2
import time
import config
from detector import MotionDetector
from logger import MotionLogger


CONFIDENCE_STYLE = {
    "HIGH": ((0, 0, 255),   "!! KRITIK HAREKET !!"),
    "LOW":  ((0, 165, 255), "? Şüpheli Hareket"),
    "NONE": (None, None),
}


def draw(frame, boxes, confidence, fps):
    color, text = CONFIDENCE_STYLE[confidence]

    if color:
        for x, y, w, h in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # FPS sol alt köşe
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


def main():
    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    if not cap.isOpened():
        print("Kamera açılamadı!")
        return

    print("Kamera açıldı! Çıkmak için 'q'ya bas.")

    detector = MotionDetector()
    logger = MotionLogger()

    prev_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Frame okunamadı!")
            break

        # FPS hesapla
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        boxes, confidence = detector.process(frame)
        draw(frame, boxes, confidence, fps)

        if confidence == "HIGH":
            logger.log(frame, boxes)

        cv2.imshow("Motion Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()