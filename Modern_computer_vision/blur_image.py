import cv2
img=cv2.imread('sim_photo.jpeg')

img_blur=cv2.GaussianBlur(img,(15,15),0) #15,15 is the kernel size and 0 is sigmax 
cv2.imshow("image",img)
cv2.imshow("grayscale image",img_blur)
cv2.waitKey(0)