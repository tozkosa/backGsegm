import cv2
import numpy as np
import matplotlib.pyplot as plt

BLUR = 51 # 21
CANNY_THRESH_1 = 6
CANNY_THRESH_2 = 20
MASK_DILATE_ITER = 10
MASK_ERODE_ITER = 10
MASK_COLOR = (0.0,0.0,1.0) # BGR

img = cv2.imread('190515-187.JPG')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
dilation = cv2.dilate(canny, None)
edges = cv2.erode(dilation, None)

# 最大の輪郭を探す
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
mask = np.zeros(dilation.shape)
cv2.fillConvexPoly(mask, max_contour[0], (255))

# 輪郭の調整
mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask

# 画像の合成
mask_stack2  = mask_stack.astype('float32') / 255.0          # Use float matrices, 
imgf         = img.astype('float32') / 255.0                  #  for easy blending

masked = (mask_stack2 * imgf) + ((1-mask_stack2) * MASK_COLOR) # Blend
masked = (masked * 255).astype('uint8')                     # Convert back to 8-bit 

c_blue, c_green, c_red = cv2.split(imgf)

img_a = cv2.merge((c_red, c_green, c_blue, mask.astype('float32') / 255.0))
img_a = (img_a * 255).astype('uint8')
print(img_a)


cv2.imwrite('cv.jpg', img)
fig = plt.figure(figsize=(15,10))
plt.subplot(2,5,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original')
plt.subplot(2,5,2)
plt.imshow(gray, cmap='gray')
plt.title('gray-scale')
plt.subplot(2,5,3)
plt.imshow(canny, cmap='gray')
plt.title('canny filter')
plt.subplot(2,5,4)
plt.imshow(dilation, cmap='gray')
plt.title('dilation')
plt.subplot(2,5,5)
plt.imshow(edges, cmap='gray')
plt.title('erode')
plt.subplot(2,5,6)
plt.imshow(mask, cmap='gray')
plt.title('Mask')
plt.subplot(2,5,7)
plt.imshow(mask_stack, cmap='gray')
plt.title('modified mask')
plt.subplot(2,5,8)
plt.imshow(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
plt.title('masked')
plt.subplot(2,5,9)
plt.imshow(img_a)
plt.show()


# kernel = np.ones((20,20), np.uint8)
# edges2 = cv2.dilate(canny, kernel, iterations=3)
# plt.subplot(2,5,6)
# plt.imshow(edges2, cmap='gray')
# plt.title('dilation with kernel')
# edges2 = cv2.erode(edges2, None)
# plt.subplot(2,5,7)
# plt.imshow(edges2, cmap='gray')


img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
cv2.imwrite('cv_after.jpg', img_a)

#print("succeeded!")