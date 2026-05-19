import csv
import cv2
from datetime import datetime
from pathlib import Path


class MotionLogger:

    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.csv_path = self.log_dir / "motion_log.csv"
        self._init_csv()

    def _init_csv(self):
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "num_objects", "boxes"])

    def log(self, frame, boxes):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # CSV'ye yaz
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, len(boxes), boxes])

        # Frame'i kaydet
        img_name = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
        img_path = self.log_dir / img_name
        cv2.imwrite(str(img_path), frame)