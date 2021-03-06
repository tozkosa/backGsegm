import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob

path = '../temp_data/*.JPG'
imgs = glob.glob(path)
print(len(imgs))

def detect_contour(img_path):
    plt.figure()
    img = cv2.imread(img_path)
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite('gray_img.jpg', gray_img)
    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(gray_img, cv2.COLOR_BGR2RGB))


    ret_, bw_img = cv2.threshold(
        gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #cv2.imwrite('bw_img.jpg', bw_img)
    plt.subplot(2, 3, 3)
    plt.imshow(cv2.cvtColor(bw_img, cv2.COLOR_BGR2RGB))

    dilation = cv2.dilate(bw_img, None, iterations=400)
    plt.subplot(2, 3, 4)
    plt.imshow(dilation, cmap='gray')
    plt.title('dilation')

    contours, h_ = cv2.findContours(
    dilation, cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    contours = list(filter(lambda x: cv2.contourArea(x) > 360000, contours))
    max = 0
    i_cont = 0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > max:
            max = area
            i_cont = i
        print(i, i_cont)
    bwc_img = cv2.drawContours(bw_img, contours, -1, (0, 255, 0), 10)
    print(len(contours))
    #cv2.imwrite('bwc_img.jpg', bwc_img)
    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(bwc_img, cv2.COLOR_BGR2RGB)) #bwc_img

    rect = cv2.boundingRect(contours[i_cont])
    crop_img = img[rect[1]:rect[1]+rect[3]-1, rect[0]:rect[0]+rect[2]-1]
    #cv2.imwrite('crop_img.jpg', crop_img)
    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    print(crop_img.shape)

    return crop_img


#detect_contour(imgs[0])

if __name__ == '__main__':

    for i in range(len(imgs)):
        words = imgs[i].split('/')[-1].split('.')[0]
        cropped = detect_contour(imgs[i])
        """ fname = f'../results/{words}.jpg'
        cv2.imwrite(fname, cropped) """
        #plt.savefig('../results/cropped'+str(i)+'.jpg')

    plt.show()

#cv2.imshow('test', gray_img)
cv2.waitKey()
