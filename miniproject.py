from ultralytics import YOLO
import cv2
import time

# โหลดโมเดล YOLO
model = YOLO("yolov8s.pt")

# เปิดกล้อง
cap = cv2.VideoCapture(1)

prev_time = 0

cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # พื้นที่ที่รถจะถอย
    zone_top = int(height * 0.4)

    cv2.rectangle(frame,
                  (0, zone_top),
                  (width, height),
                  (255,255,0), 2)

    # -------- GRID 3x3 --------
    rows = 3
    cols = 3

    cell_w = width // cols
    cell_h = (height - zone_top) // rows

    for i in range(1, cols):
        x = i * cell_w
        cv2.line(frame,(x,zone_top),(x,height),(255,255,0),1)

    for j in range(1, rows):
        y = zone_top + j * cell_h
        cv2.line(frame,(0,y),(width,y),(255,255,0),1)

    # -------- Danger Zone --------
    danger_line = int(height * 0.7)
    cv2.line(frame,(0,danger_line),(width,danger_line),(0,0,255),3)

    # -------- ROI (หลังรถ) --------
    roi = frame[zone_top:height, 0:width]

    # ตรวจจับวัตถุ
    results = model(roi, imgsz=960, conf=0.5, stream=True)

    person_count = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "person":

                person_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # ปรับตำแหน่งจาก ROI
                y1 += zone_top
                y2 += zone_top

                conf = float(box.conf[0])

                # สีกรอบ
                color = (0,255,0)

                if y2 > danger_line:
                    color = (0,0,255)

                    cv2.putText(frame,
                                "WARNING PERSON BEHIND CAR",
                                (50,120),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0,0,255),
                                3)

                # วาดกรอบ
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

                cv2.putText(frame,
                            f"Person {conf:.2f}",
                            (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0,255,0),
                            2)

    # -------- จำนวนคน --------
    cv2.putText(frame,
                f"Total People: {person_count}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                3)

    # -------- FPS --------
    current_time = time.time()
    fps = 1/(current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    cv2.putText(frame,
                f"FPS: {int(fps)}",
                (20,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,0,0),
                3)

    cv2.imshow("Real-Time Person Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()