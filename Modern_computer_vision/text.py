import cv2
img=cv2.imread('sim_photo.jpeg')

cv2.putText(img,"hello this is simran",(350,100),cv2.FONT_HERSHEY_DUPLEX,2,(200,0,200),2)
#img,text,position,font,fontsize,color,thickness
cv2.imshow("image",img)

cv2.waitKey(0)