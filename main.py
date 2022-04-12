# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2 as cv
import numpy as np

#C:\Users\BradC\Documents\University Folder\HotAirBalloons.jpg
def blur_image():
    img = cv.imread('C:\\Users\\BradC\\Documents\\University Folder\\Major Project\\HotAirBalloons.jpg')

    gaussian_blur = cv.GaussianBlur(img, (31, 31), 0)

    gaussian_blur2 = cv.GaussianBlur(gaussian_blur, (31, 31), 0)

    bilateral_blur = cv.bilateralFilter(gaussian_blur2, 35, 300, 300)

    bilateral_blur_unedited = cv.bilateralFilter(img, 35, 300, 300)

    #cv.imwrite('C:\\Users\\BradC\\Documents\\University Folder\\Major Project\\HotAirBalloonsGaussThenBilateral.jpg', bilateral_blur)
    #
    # cv.imwrite('C:\\Users\\BradC\\Documents\\University Folder\\Major Project\\HotAirBalloonsGaussThenGauss.jpg',
    #            gaussian_blur2)
    # cv.imwrite('C:\\Users\\BradC\\Documents\\University Folder\\Major Project\\HotAirBalloonsGauss.jpg',
    #            gaussian_blur)
    #
    # cv.imwrite('C:\\Users\\BradC\\Documents\\University Folder\\Major Project\\HotAirBalloonsGaussGaussBilateral.jpg',
    #            bilateral_blur)
    #
    # cv.imwrite('C:\\Users\\BradC\\Documents\\University Folder\\Major Project\\HotAirBalloonsIntenseBilateral.jpg',
    #            bilateral_blur_unedited)


def quant_image(image, k):
    i = np.float32(image).reshape(-1, 3)
    condition = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,20,1.0)
    ret,label,center = cv.kmeans(i, k , None, condition,10,cv.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)

    final_img = center[label.flatten()]
    final_img = final_img.reshape(image.shape)
    return final_img






# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #blur_image()

    storage_location = 'C:\\Users\\BradC\\Documents\\University Folder\\Major Project\\Images Currently Produced\\'

    img = cv.imread('C:\\Users\\BradC\\Documents\\University Folder\\Major Project\\LakeView.jpg')
    #img = cv.bilateralFilter(img, 35, 75, 75)
    quantizedImage = quant_image(img, 8)
    cv.imwrite(storage_location + 'Bilateral75ColorQuantizationLakeView.jpg',quantizedImage)

    quantizedImageGrey = cv.cvtColor(quantizedImage, cv.COLOR_BGR2GRAY)

    cv.imwrite(str(storage_location) + 'LakeViewValueStudy.jpg', quantizedImageGrey)


    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
