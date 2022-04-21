# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2 as cv
import numpy as np


def quant_image(image, k):
    i = np.float32(image).reshape(-1, 3)
    condition = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    ret, label, center = cv.kmeans(i, k, None, condition, 10, cv.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)

    final_img = center[label.flatten()]
    final_img = final_img.reshape(image.shape)
    return final_img


def retrieve_value_study(image):
    white_threshold = 220
    light_threshold = 180
    mid_threshold = 145
    dark_threshold = 90

    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j][2] > white_threshold:
                image[i][j][2] = 255
            elif image[i][j][2] > light_threshold:
                image[i][j][2] = light_threshold
            elif image[i][j][2] > mid_threshold:
                image[i][j][2] = mid_threshold
            elif image[i][j][2] > dark_threshold:
                image[i][j][2] = dark_threshold
            else:
                image[i][j][2] = 40

    return image


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    storage_location = 'resources\\produced_images\\'

    img = cv.imread('resources\\test_images\\LakeView.jpg')
    img = cv.bilateralFilter(img, 35, 75, 75)
    quantizedImage = quant_image(img, 8)
    # cv.imwrite(storage_location + 'ColorQuantizationLakeView.jpg',quantizedImage)

    quantisedImageHSV = cv.cvtColor(quantizedImage, cv.COLOR_BGR2HSV)
    value_study_test = retrieve_value_study(quantisedImageHSV)



    value_study_test = cv.cvtColor(value_study_test, cv.COLOR_HSV2BGR)
    value_study_black_and_white_test = cv.cvtColor(value_study_test, cv.COLOR_BGR2GRAY)

    cv.imwrite(storage_location + 'test.jpg', value_study_test)
    cv.imwrite(storage_location + 'testBW.jpg', value_study_black_and_white_test)

    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
