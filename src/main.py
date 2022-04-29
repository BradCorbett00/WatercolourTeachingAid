import glob

import PySimpleGUI
import cv2 as cv
import numpy as np
import PySimpleGUI as gui
import os.path
from PIL import Image
from PIL import ImageTk


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

    image_whites_and_lights = image.copy()
    image_whites_lights_and_mids = image.copy()
    image_whites_lights_mids_and_darks = image.copy()
    value_study_image = image.copy()



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


def resize_opencv_image(img, target_dimension):  # Works with an OpenCV image
    width = int(img.shape[1])
    height = int(img.shape[0])

    new_size = reduce_resolution(height, width, target_dimension)

    new_img = cv.resize(img, new_size, interpolation=cv.INTER_AREA)

    return new_img


def resize_pil_image(image_to_shrink, target_dimension) -> tuple[int, int]:  # Works with a PIL image
    width = int(image_to_shrink.size[0])
    height = int(image_to_shrink.size[1])

    new_size = reduce_resolution(height, width, target_dimension)

    return new_size


def reduce_resolution(height, width, target_dimension):
    image_below_target: bool = ((width < target_dimension) and (height < target_dimension))
    while not image_below_target:
        width = int(width * 0.75)
        height = int(height * 0.75)
        image_below_target = ((width <= target_dimension) and (height <= target_dimension))
    new_size = (width, height)
    return new_size


def process_image(img, target_dimension):
    img = resize_opencv_image(img, target_dimension)

    img = cv.bilateralFilter(img, 35, 75, 75)

    img = quant_image(img, 8)

    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    value_study_list_HSV = retrieve_value_study(img)

    value_study_lights = cv.cvtColor(value_study_list_HSV[0], cv.COLOR_HSV2BGR)
    value_study_mids = cv.cvtColor(value_study_list_HSV[1], cv.COLOR_HSV2BGR)
    value_study_darks = cv.cvtColor(value_study_list_HSV[2], cv.COLOR_HSV2BGR)
    value_study_list_full_all_tones = cv.cvtColor(value_study_list_HSV[3], cv.COLOR_HSV2BGR)
    value_study_black_and_white = cv.cvtColor(value_study_list_full_all_tones, cv.COLOR_BGR2GRAY)

    value_study_list = [value_study_lights, value_study_mids, value_study_darks, value_study_list_full_all_tones,
                        value_study_black_and_white]

    return value_study_list


def save_processed_image(value_study_list, filename):
    filename_base = os.path.basename(filename)

    storage_location = 'resources\\produced_images\\' + '\\' + filename_base

    if not (os.path.exists(storage_location)):
        os.makedirs(storage_location)

    value_study_lights = value_study_list[0]
    value_study_mids = value_study_list[1]
    value_study_darks = value_study_list[2]
    value_study_all_tones = value_study_list[3]
    value_study_black_and_white = value_study_list[4]

    cv.imwrite(storage_location + '\\' + 'light_value_study.png', value_study_lights)
    cv.imwrite(storage_location + '\\' + 'mid_value_study.png', value_study_mids)
    cv.imwrite(storage_location + '\\' + 'dark_value_study.png', value_study_darks)
    cv.imwrite(storage_location + '\\' + 'full_value_study.png', value_study_all_tones)
    cv.imwrite(storage_location + '\\' + 'black_white_value_study.png',
               value_study_black_and_white)


def create_ui():
    file_selector = [
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

    image_to_process_column = [
        [gui.Text("Choose an image from list on left:")],
        [gui.Text(size=(40, 1), key="-TOUT-")],
        [gui.Image(key="-IMAGE-")],
        [gui.Button("Process This Photo?", key="-BUTTON-", visible=False)]
    ]

    painting_layout = [
        [gui.Text("Current Step", key="-STEP_TRACKER-")],
        [gui.Image(key="-CURRENT_STEP-")],
        [gui.Button("Previous Step", key="-PREVIOUS-", visible=False)],
        [gui.Button("Next Step", key="-NEXT-", visible=True)]
    ]

    image_selection_layout = [
        [
            gui.Column(file_selector),
            gui.VSeperator(),
            gui.Column(image_to_process_column),
        ]
    ]

    final_layout = [
        [
            gui.Column(image_selection_layout, key="-IMAGE_SELECTION-", visible=True),
            gui.Column(painting_layout, key="-PAINTING-", visible=False)
        ]
    ]

    display_window = gui.Window("Tech-Nicolour", final_layout, resizable=True)

    images = []
    current_picture = 0

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
            filename = os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0]
            )
            im = Image.open(filename)
            from PIL.Image import Resampling

            size = resize_pil_image(im, 720)

            im = im.resize(size, resample=Resampling.BICUBIC)

            image = ImageTk.PhotoImage(image=im)

            display_window["-TOUT-"].update(filename)
            display_window["-IMAGE-"].update(data=image)
            display_window["-BUTTON-"].update(visible=True)
        elif event == "-BUTTON-":
            display_window["-IMAGE_SELECTION-"].update(visible=False)
            display_window["-PAINTING-"].update(visible=True)

            images.clear()
            img = cv.imread(filename)

            value_study_list = process_image(img, 1280)

            # save_processed_image(value_study_list, filename)

            filename_base = os.path.basename(filename)


            for image in value_study_list:
                temp_im = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                temp_im = Image.fromarray(temp_im)
                temp_im = ImageTk.PhotoImage(temp_im)
                images.append(temp_im)

            # for image in os.listdir('resources\\produced_images\\' + filename_base):
            #     temp_im = Image.open('resources\\produced_images\\' + filename_base + '\\' + image)
            #     temp_im = ImageTk.PhotoImage(temp_im)
            #     images.append(temp_im)

            display_window["-CURRENT_STEP-"].update(data=images[current_picture])

        elif event == "-NEXT-":
            current_picture = current_picture + 1
            display_window["-CURRENT_STEP-"].update(data=images[current_picture])
            display_window["-PREVIOUS-"].update(visible=True)
            if current_picture >= len(images) - 1:
                display_window["-NEXT-"].update(visible=False)

        elif event == "-PREVIOUS-":
            current_picture = current_picture - 1
            display_window["-CURRENT_STEP-"].update(data=images[current_picture])
            display_window["-NEXT-"].update(visible=True)
            if current_picture == 0:
                display_window["-PREVIOUS-"].update(visible=False)

    display_window.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    create_ui()
    # storage_location = 'resources\\produced_images\\'
    # img = cv.imread('resources\\test_images\\Glacier.jpg')
    #
    # img_name = "Glacier"
    #
    # value_study_test = process_image(img, 1280)
    #
    # if not (os.path.exists(storage_location + '\\' + img_name)):
    #     os.makedirs(storage_location + '\\' + img_name)
    #
    # value_study_lights = cv.cvtColor(value_study_test[0], cv.COLOR_HSV2BGR)
    # value_study_mids = cv.cvtColor(value_study_test[1], cv.COLOR_HSV2BGR)
    # value_study_darks = cv.cvtColor(value_study_test[2], cv.COLOR_HSV2BGR)
    # value_study_test_full = cv.cvtColor(value_study_test[3], cv.COLOR_HSV2BGR)
    # value_study_black_and_white_test = cv.cvtColor(value_study_test_full, cv.COLOR_BGR2GRAY)
    #
    # cv.imwrite(storage_location + '\\' + img_name + '\\' + 'light_value_study.png', value_study_lights)
    # cv.imwrite(storage_location + '\\' + img_name + '\\' + 'mid_value_study.png', value_study_mids)
    # cv.imwrite(storage_location + '\\' + img_name + '\\' + 'dark_value_study.png', value_study_darks)
    # cv.imwrite(storage_location + '\\' + img_name + '\\' + 'full_value_study.png', value_study_test_full)
    # cv.imwrite(storage_location + '\\' + img_name + '\\' + 'black_white_value_study.png',
    #            value_study_black_and_white_test)
