import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('190515-187.JPG')
plt.subplot(2,3,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray_img.jpg', gray_img)

plt.subplot(2,3,2)
plt.imshow(cv2.cvtColor(gray_img, cv2.COLOR_BGR2RGB))


ret_, bw_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imwrite('bw_img.jpg', bw_img)
plt.subplot(2,3,3)
plt.imshow(cv2.cvtColor(bw_img, cv2.COLOR_BGR2RGB))

contours, h_ = cv2.findContours(bw_img, cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
bwc_img = cv2.drawContours(bw_img, contours, -1, (0,255,0), 10)
cv2.imwrite('bwc_img.jpg', bwc_img)
plt.subplot(2,3,4)
plt.imshow(cv2.cvtColor(bwc_img, cv2.COLOR_BGR2RGB))

rect = cv2.boundingRect(contours[-1])
crop_img = img[rect[1]:rect[1]+rect[3]-1, rect[0]:rect[0]+rect[2]-1]
cv2.imwrite('crop_img.jpg', crop_img)
plt.subplot(2,3,5)
plt.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
print(crop_img.shape)

plt.show()

#cv2.imshow('test', gray_img)
#cv2.waitKey()