import cv2
img=cv2.imread('sim_photo.jpeg')

img_crop=img[100:500,100:200] #[y1:y2,x1:x2]
cv2.imshow("image",img)
cv2.imshow("grayscale image",img_crop)
cv2.waitKey(0)