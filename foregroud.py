import numpy as np
import cv2

img = cv2.imread('180606-047.JPG')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray_img.jpg', gray_img)
ret_, bw_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imwrite('bw_img.jpg', bw_img)
contours, h_ = cv2.findContours(bw_img, cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
bwc_img = cv2.drawContours(bw_img, contours, -1, (0,255,0), 10)
cv2.imwrite('bwc_img.jpg', bwc_img)
rect = cv2.boundingRect(contours[-1])
crop_img = img[rect[1]:rect[1]+rect[3]-1, rect[0]:rect[0]+rect[2]-1]
cv2.imwrite('crop_img.jpg', crop_img)

#cv2.imshow('test', gray_img)
#cv2.waitKey()