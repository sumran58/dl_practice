import cv2
img=cv2.imread('sim_photo.jpeg')

img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("image",img)
cv2.imshow("grayscale image",img_gray)
cv2.waitKey(0)