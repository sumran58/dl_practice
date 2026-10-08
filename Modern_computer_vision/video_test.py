import cv2
cap=cv2.VideoCapture('Men_tailoring_garments_in_shop_202609081404.mp4')

while True:
    success , img = cap.read()
    if not success:
        break
    cv2.imshow('tailoring video',img)
    cv2.waitKey(1)