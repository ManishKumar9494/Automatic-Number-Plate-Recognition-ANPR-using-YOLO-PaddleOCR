import cv2
import os
import re
import time
from datetime import datetime
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Load models
model = YOLO("license.pt")   # Replace with fine-tuned plate model if available
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Save folder
os.makedirs("plates", exist_ok=True)

# Track last detections
last_saved = {}  # {plate_text: timestamp}


def clean_plate_text(ocr_result):
    plate_text = ""
    if ocr_result and len(ocr_result[0]) > 0:
        parts = []
        for line in ocr_result[0]:
            text = line[1][0].upper().replace(" ", "")
            if text.isalnum() and len(text) >= 2:
                parts.append(text)
        candidate = "".join(parts)

        # Regex for Indian number plate
        pattern = r"[A-Z]{2}[0-9]{2}[A-Z0-9]{1,2}[0-9]{4}"
        match = re.search(pattern, candidate)
        if match:
            plate_text = match.group(0)
        else:
            plate_text = candidate
    return plate_text


def run_anpr_camera():
    cap = cv2.VideoCapture("add camera ip")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)

            for box in boxes:
                x1, y1, x2, y2 = box[:4]
                plate_crop = frame[y1:y2, x1:x2]

                if plate_crop.size != 0:
                    ocr_result = ocr.ocr(plate_crop, cls=True)
                    detected_text = clean_plate_text(ocr_result)

                    if detected_text:
                        print(f"✅ Final Plate: {detected_text}")
                        cv2.putText(frame, detected_text, (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Save only if new or after cooldown
                        now = time.time()
                        cooldown = 8  # seconds
                        if (detected_text not in last_saved) or (now - last_saved[detected_text] > cooldown):
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            cv2.imwrite(f"plates/{detected_text}_{timestamp}.jpg", plate_crop)
                            with open("plates/detected_plates.txt", "a") as f:
                                f.write(f"{timestamp} - {detected_text}\n")
                            last_saved[detected_text] = now  # update timestamp

        cv2.imshow("ANPR Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_anpr_camera()
