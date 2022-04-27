# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import time

import cv2
import cv2 as cv
import numpy as np
import PySimpleGUI as gui
import os.path


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
    really_dark_value = 40

    for i in range(
            len(image)):  # Turns the image into lights and whites. The non-white/lights are slightly darker to signify for later.
        for j in range(len(image[i])):
            if image[i][j][2] > white_threshold:
                image[i][j][2] = 255
            elif image[i][j][2] > light_threshold:
                image[i][j][2] = light_threshold
            elif image[i][j][2] > mid_threshold:
                image[i][j][2] = light_threshold - 1  # A mid
            elif image[i][j][2] > dark_threshold:
                image[i][j][2] = light_threshold - 2  # A dark
            else:
                image[i][j][2] = light_threshold - 3  # A really dark nearly black

    image_whites_and_lights = image.copy()

    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j][2] >= light_threshold:
                continue
            elif image[i][j][2] == light_threshold - 1:
                image[i][j][2] = mid_threshold
            elif image[i][j][2] == light_threshold - 2:
                image[i][j][2] = mid_threshold - 1  # A dark
            else:
                image[i][j][2] = mid_threshold - 2  # A really dark

    image_whites_lights_and_mids = image.copy()

    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j][2] >= mid_threshold:
                continue
            elif image[i][j][2] == mid_threshold - 1:
                image[i][j][2] = dark_threshold
            elif image[i][j][2] == mid_threshold - 2:
                image[i][j][2] = dark_threshold - 1  # A really dark

    image_whites_lights_mids_and_darks = image.copy()

    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j][2] < dark_threshold:
                image[i][j][2] = really_dark_value

    value_study_image = image

    return image_whites_and_lights, image_whites_lights_and_mids, image_whites_lights_mids_and_darks, value_study_image


def resize_image(img, scale):
    new_width = int(img.shape[1] * scale / 100)
    new_height = int(img.shape[0] * scale / 100)
    new_size = (1280, 720)
    new_img = cv.resize(img, new_size, interpolation=cv.INTER_AREA)
    return new_img


def process_image(img, scale):
    img = resize_image(img, 50)

    img = cv.bilateralFilter(img, 35, 75, 75)

    img = quant_image(img, 8)

    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    value_study_list = retrieve_value_study(img)

    return value_study_list


def create_ui():
    file_list_column = [
        [
            gui.Text("Image Source"),
            gui.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
            gui.FolderBrowse(),
        ],
        [
            gui.Listbox(
                values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
            )
        ],
    ]

    image_viewer_column = [
        [gui.Text("Choose an image from list on left:")],
        [gui.Text(size=(40, 1), key="-TOUT-")],
        [gui.Image(key="-IMAGE-")],
    ]

    layout = [
        [
            gui.Column(file_list_column),
            gui.VSeperator(),
            gui.Column(image_viewer_column),
        ]
    ]

    display_window = gui.Window("Tech-Nicolour", layout)

    while True:
        event, values = display_window.read()
        if event == "Exit" or event == gui.WIN_CLOSED:
            break

        if event == "-FOLDER-":
            folder = values["-FOLDER-"]
            try:
                # Gets all files within the folder
                file_list = os.listdir(folder)
            except:
                file_list = []
            file_names = [
                f
                for f in file_list
                if os.path.isfile(os.path.join(folder, f))
                   and f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            display_window["-FILE LIST-"].update(file_names)
        elif event == "-FILE LIST-":  # A file was chosen from the listbox
            try:
                filename = os.path.join(
                    values["-FOLDER-"], values["-FILE LIST-"][0]
                )
                display_window["-TOUT-"].update(filename)
                display_window["-IMAGE-"].update(filename=filename)
            except:
                pass

    display_window.close()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    create_ui()
    storage_location = 'resources\\produced_images\\'

    startTime = time.time()
    img = cv.imread('resources\\test_images\\Glacier.jpg')
    readTime = time.time() - startTime

    img_name = "Glacier"

    value_study_test = process_image(img, 50)

    value_study_lights = cv.cvtColor(value_study_test[0], cv.COLOR_HSV2BGR)
    value_study_mids = cv.cvtColor(value_study_test[1], cv.COLOR_HSV2BGR)
    value_study_darks = cv.cvtColor(value_study_test[2], cv.COLOR_HSV2BGR)
    value_study_test_full = cv.cvtColor(value_study_test[3], cv.COLOR_HSV2BGR)
    value_study_black_and_white_test = cv.cvtColor(value_study_test_full, cv.COLOR_BGR2GRAY)

    cv.imwrite(storage_location + img_name + '_lightValueStudy.jpg', value_study_lights)
    cv.imwrite(storage_location + img_name + '_midValueStudy.jpg', value_study_mids)
    cv.imwrite(storage_location + img_name + '_darkValueStudy.jpg', value_study_darks)
    cv.imwrite(storage_location + img_name + '_fullValueStudy.jpg', value_study_test_full)
    cv.imwrite(storage_location + img_name + '_testBW.jpg', value_study_black_and_white_test)
