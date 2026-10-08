import cv2
cap=cv2.VideoCapture(0) # 0 is the id of the laptop camera 
 
while True:
    success , img = cap.read()
    if not success:
        break
    cv2.imshow('tailoring video',img)
    if cv2.waitKey(1) & 0xFF ==ord('q'): #q press karege ti band hoega webcam 
        break 
    