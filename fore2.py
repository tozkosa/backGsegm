import cv2
import numpy as np
import matplotlib.pyplot as plt

BLUR = 21
CANNY_THRESH_1 = 20
CANNY_THRESH_2 = 90  # 100
MASK_DILATE_ITER = 10
MASK_ERODE_ITER = 10
MASK_COLOR = (0.0,0.0,1.0) # In BGR format

img = cv2.imread('190521-119.jpg')
cv2.imwrite('cv.jpg', img)

plt.figure()
plt.subplot(2,3,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

plt.subplot(2,3,2)
plt.imshow(gray, cmap='gray')


edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
edges = cv2.dilate(edges, None)
edges = cv2.erode(edges, None)

plt.subplot(2,3,3)
plt.imshow(edges, cmap='gray')


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

mask = np.zeros(edges.shape)
cv2.fillConvexPoly(mask, max_contour[0], (255))

plt.subplot(2,3,4)
plt.imshow(mask, cmap='gray')

mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask

plt.subplot(2,3,5)
plt.imshow(mask_stack, cmap='gray')

mask_stack  = mask_stack.astype('float32') / 255.0          # Use float matrices, 
img         = img.astype('float32') / 255.0                 #  for easy blending

masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR) # Blend
masked = (masked * 255).astype('uint8')                     # Convert back to 8-bit 

c_blue, c_green, c_red = cv2.split(img)

img_a = cv2.merge((c_red, c_green, c_blue, mask.astype('float32') / 255.0))

plt.subplot(2,3,6)
plt.imshow(img_a)
plt.show()