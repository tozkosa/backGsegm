import cv2
import numpy as np
import matplotlib.pyplot as plt

BLUR = 21
CANNY_THRESH_1 = 3
CANNY_THRESH_2 = 10
MASK_DILATE_ITER = 10
MASK_ERODE_ITER = 10
MASK_COLOR = (0.0,0.0,1.0) # BGR

img = cv2.imread('../selby_labelled_data/low/191008-007_1.png')
cv2.imwrite('cv.jpg', img)

plt.figure(figsize=(15,10))
plt.subplot(2,5,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.subplot(2,5,2)
plt.imshow(gray, cmap='gray')

edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
plt.subplot(2,5,3)
plt.imshow(edges, cmap='gray')

edges = cv2.dilate(edges, None)
plt.subplot(2,5,4)
plt.imshow(edges, cmap='gray')
kernel = np.ones((5,5), np.uint8)
edges2 = cv2.dilate(edges, kernel, iterations=3)
plt.subplot(2,5,5)
plt.imshow(edges2, cmap='gray')
edges2 = cv2.erode(edges2, None)
plt.subplot(2,5,6)
plt.imshow(edges2, cmap='gray')

contour_info = []
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

for c in contours:
    contour_info.append((
        c,
        cv2.isContourConvex(c),
        cv2.contourArea(c),
    ))
contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
max_contour = contour_info[0]

#-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
# Mask is black, polygon is white
mask = np.zeros(edges.shape)
cv2.fillConvexPoly(mask, max_contour[0], (255))

plt.subplot(2,5,7)
plt.imshow(mask, cmap='gray')

mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask

plt.subplot(2,5,8)
plt.imshow(mask_stack, cmap='gray')

mask_stack  = mask_stack.astype('float32') / 255.0          # Use float matrices, 
img         = img.astype('float32') / 255.0                 #  for easy blending

masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR) # Blend
masked = (masked * 255).astype('uint8')                     # Convert back to 8-bit 

c_blue, c_green, c_red = cv2.split(img)

img_a = cv2.merge((c_red, c_green, c_blue, mask.astype('float32') / 255.0))

plt.subplot(2,5, 9)
plt.imshow(img_a)
plt.show()

img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
cv2.imwrite('cv_after.jpg', (img_a * 255).astype('uint8'))



#print("succeeded!")