import cv2
import time
import pygame

# -------- ตั้งค่าระบบเสียง --------
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("Magic.wav") 

# เปิดกล้อง
cap = cv2.VideoCapture(0)

prev_time = 0
cap.set(cv2.CAP_PROP_FPS, 30)

# สร้างตัวตรวจจับการเคลื่อนไหว (Background Subtractor)
# history คือจำนวนเฟรมที่ใช้จำพื้นหลัง, varThreshold คือความไวต่อการเปลี่ยนแปลง (ยิ่งน้อยยิ่งไว)
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=False)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    zone_top = int(height * 0.4)

    # วาดกรอบพื้นที่ ROI
    cv2.rectangle(frame, (0, zone_top), (width, height), (255, 255, 0), 2)

    # -------- GRID 3x3 --------
    rows = 3
    cols = 3
    cell_w = width // cols
    cell_h = (height - zone_top) // rows

    for i in range(1, cols):
        x = i * cell_w
        cv2.line(frame, (x, zone_top), (x, height), (255, 255, 0), 1)

    for j in range(1, rows):
        y = zone_top + j * cell_h
        cv2.line(frame, (0, y), (width, y), (255, 255, 0), 1)

    # -------- Danger Zone --------
    danger_line = int(height * 0.7)
    cv2.line(frame, (0, danger_line), (width, danger_line), (0, 0, 255), 3)

    # -------- ROI (หลังรถ) --------
    roi = frame[zone_top:height, 0:width]

    # นำ ROI ไปเข้าสู่กระบวนการหาการเคลื่อนไหว (จะได้ภาพขาวดำ พื้นหลังดำ วัตถุขยับเป็นสีขาว)
    mask = bg_subtractor.apply(roi)

    # กรองลบนอยส์ (จุดขาวเล็กๆ ที่เกิดจากแสงกระพริบ) ออกไป
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    mask = cv2.medianBlur(mask, 5)

    # หาเส้นขอบ (Contours) ของวัตถุที่เคลื่อนไหว
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ตัวแปรเช็คว่ามีวัตถุในโซนอันตรายหรือไม่
    object_in_danger_zone = False 
    object_count = 0

    for contour in contours:
        # กรองขนาด ถ้าพื้นที่เคลื่อนไหวน้อยกว่า 1000 พิกเซล ให้มองข้ามไป (ป้องกันใบไม้ไหว)
        if cv2.contourArea(contour) > 1000:
            object_count += 1
            
            # ตีกรอบสี่เหลี่ยมคลุมวัตถุที่ขยับ
            x, y, w, h = cv2.boundingRect(contour)

            # ปรับพิกัดแกน y ให้ตรงกับภาพใหญ่เต็มจอ
            x1 = x
            y1 = y + zone_top
            x2 = x + w
            y2 = y + zone_top + h

            color = (0, 255, 255) # สีเหลืองสำหรับวัตถุที่ตรวจจับได้

            # -------- เมื่อวัตถุล้ำเส้น Danger Line --------
            if y2 > danger_line:
                color = (0, 0, 255)
                cv2.putText(frame, "WARNING: OBJECT DETECTED!", (50, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                object_in_danger_zone = True

            # วาดกรอบสี่เหลี่ยม
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, "Moving Object", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # -------- จัดการระบบเสียงเตือน --------
    if object_in_danger_zone:
        if not pygame.mixer.get_busy():
            alert_sound.play()
    else:
        alert_sound.stop()

    # -------- จำนวนวัตถุที่เคลื่อนไหว --------
    cv2.putText(frame, f"Objects: {object_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # -------- FPS --------
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow("Real-Time Motion Detection", frame)
    
    # ดูภาพ Mask (ภาพขาวดำที่คอมพิวเตอร์เห็น) ไว้สำหรับปรับจูนความแม่นยำ - เอาเครื่องหมาย # ออกถ้าอยากดู
    # cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()