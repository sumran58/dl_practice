import cv2
img=cv2.imread('sim_photo.jpeg')
cv2.imshow("my photo",img)
cv2.waitKey(0) #0 means kep it open until we press anything and if other than 0 means 1000 means keep for 1 sec