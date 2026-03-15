import cv2
import time
import pygame

# -------- 1. ตั้งค่าระบบเสียง (ห้ามลบ) --------
pygame.mixer.init()
try:
    alert_sound = pygame.mixer.Sound("Magic.wav") 
except:
    print("Warning: ไม่พบไฟล์ Magic.wav")
    alert_sound = None

# -------- 2. เปิดกล้อง (ใช้เลข 0 ตามที่คุณเช็คเจอ) --------
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FPS, 30)
prev_time = 0

# -------- 3. สร้างตัวตรวจจับความเคลื่อนไหว (แทนที่ YOLO) --------
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=False)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    zone_top = int(height * 0.4) # เริ่มตรวจจับที่ 40% ของจอ

    # -------- 4. Danger Zone (เส้นแดง 70%) --------
    danger_line = int(height * 0.7)
    cv2.line(frame, (0, danger_line), (width, danger_line), (0, 0, 255), 3)
    
    # -------- 5.Parking Guide Lines --------
    bottom_y = height
    top_y = int(height * 0.6)

    # เส้นซ้าย
    cv2.line(frame, (int(width*0.25), height), (int(width*0.45), int(height*0.6)), (0,255,0), 4)

    # เส้นขวา
    cv2.line(frame, (int(width*0.75), height), (int(width*0.55), int(height*0.6)), (0,255,0), 4)

    # เส้นระยะ
    cv2.line(frame, (int(width*0.42), int(height*0.8)), (int(width*0.58), int(height*0.8)), (0,255,255), 3)
    cv2.line(frame, (int(width*0.40), int(height*0.9)), (int(width*0.60), int(height*0.9)), (0,0,255), 3)

    # -------- 6. ประมวลผลภาพเพื่อหาการขยับ --------
    roi = frame[zone_top:height, 0:width]
    mask = bg_subtractor.apply(roi)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    mask = cv2.medianBlur(mask, 5)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    object_in_danger_zone = False 
    object_count = 0

    for contour in contours:
        if cv2.contourArea(contour) > 1000: # กรองขนาดวัตถุ
            object_count += 1
            x, y, w, h = cv2.boundingRect(contour)
            
            # พิกัดของวัตถุ
            x1, y1 = x, y + zone_top
            x2, y2 = x + w, y + zone_top + h

            color = (0, 255, 255) # สีเหลืองปกติ

            # เช็คว่าล้ำเส้นแดงไหม
            if y2 > danger_line:
                color = (0, 0, 255) # เปลี่ยนเป็นสีแดง
                cv2.putText(frame, "WARNING!", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                object_in_danger_zone = True

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # -------- 7. ระบบเสียงเตือน --------
    if alert_sound:
        if object_in_danger_zone:
            if not pygame.mixer.get_busy():
                alert_sound.play()
        else:
            alert_sound.stop()

    # -------- 8. แสดงผล FPS และสถานะ --------
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time
    cv2.putText(frame, f"Objects: {object_count} | FPS: {int(fps)}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Rear-view (Motion Only - No YOLO)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()