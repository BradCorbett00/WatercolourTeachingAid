import concurrent.futures
import cv2 as cv
import numpy
import numpy as np
import PySimpleGUI as gui
import os.path
from PIL import Image
from PIL import ImageDraw
from PIL import ImageTk


def produce_sketch(image):
    grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    inverted = cv.bitwise_not(grey)

    blurred = cv.GaussianBlur(inverted, (51, 51), 0)

    inverted_blur = cv.bitwise_not(blurred)

    final_sketch = cv.divide(grey, inverted_blur, scale=256.0)

    return final_sketch


def reduce_colours_and_produce_palette(image, k):
    reshaped_image = np.float32(image).reshape(-1, 3)
    condition = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    ret, label, center = cv.kmeans(reshaped_image, k, None, condition, 10, cv.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)

    final_img = center[label.flatten()]

    final_img = final_img.reshape(image.shape)

    # Palette Generation
    colours = center
    n = len(colours)
    im = Image.new('RGB', (100 * n, 100))
    draw = ImageDraw.Draw(im)
    for index, color in enumerate(colours):
        color = tuple([int(x) for x in color])
        draw.rectangle([(100 * index, 0), (100 * (index + 1), 100 * (index + 1))],
                       fill=tuple(color))

    return final_img, im


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

    highlight_lights_image = image.copy()
    highlight_mids_image = image.copy()
    highlight_darks_image = image.copy()
    highlight_really_darks_image = image.copy()

    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j][2] > white_threshold and image[i][j][1] < 100:  # If low saturation, then white
                image_whites_and_lights[i][j][1] = 0  # Desaturates
                image_whites_and_lights[i][j][2] = 255  # Maximises the value

                image_whites_lights_and_mids[i][j][1] = 0
                image_whites_lights_and_mids[i][j][2] = 255

                image_whites_lights_mids_and_darks[i][j][1] = 0
                image_whites_lights_mids_and_darks[i][j][2] = 255

                value_study_image[i][j][1] = 0
                value_study_image[i][j][2] = 255

                highlight_lights_image[i][j][1] = 0
                highlight_lights_image[i][j][2] = 255

                highlight_mids_image[i][j][1] = 0
                highlight_mids_image[i][j][2] = 255

                highlight_darks_image[i][j][1] = 0
                highlight_darks_image[i][j][2] = 255

                highlight_really_darks_image[i][j][1] = 0
                highlight_really_darks_image[i][j][2] = 255

            elif image[i][j][2] > light_threshold:
                image_whites_and_lights[i][j][2] = light_threshold
                image_whites_lights_and_mids[i][j][2] = light_threshold
                image_whites_lights_mids_and_darks[i][j][2] = light_threshold
                value_study_image[i][j][2] = light_threshold

                highlight_mids_image[i][j][2] = light_threshold
                highlight_darks_image[i][j][2] = light_threshold
                highlight_really_darks_image[i][j][2] = light_threshold

                # Highlighted Pixels
                highlight_lights_image[i][j][1] = 255
                highlight_lights_image[i][j][2] = 255

            elif image[i][j][2] > mid_threshold:
                image_whites_and_lights[i][j][2] = light_threshold
                image_whites_lights_and_mids[i][j][2] = mid_threshold
                image_whites_lights_mids_and_darks[i][j][2] = mid_threshold
                value_study_image[i][j][2] = mid_threshold

                highlight_darks_image[i][j][2] = mid_threshold
                highlight_really_darks_image[i][j][2] = mid_threshold

                # Highlighted Pixels
                highlight_lights_image[i][j][1] = 255
                highlight_lights_image[i][j][2] = 255

                highlight_mids_image[i][j][1] = 255
                highlight_mids_image[i][j][2] = 255

            elif image[i][j][2] > dark_threshold:
                image_whites_and_lights[i][j][2] = light_threshold
                image_whites_lights_and_mids[i][j][2] = mid_threshold
                image_whites_lights_mids_and_darks[i][j][2] = dark_threshold
                value_study_image[i][j][2] = dark_threshold

                highlight_really_darks_image[i][j][2] = dark_threshold

                # Highlighted Pixels
                highlight_lights_image[i][j][1] = 255
                highlight_lights_image[i][j][2] = 255

                highlight_mids_image[i][j][1] = 255
                highlight_mids_image[i][j][2] = 255

                highlight_darks_image[i][j][1] = 255
                highlight_darks_image[i][j][2] = 255

            else:
                image_whites_and_lights[i][j][2] = light_threshold
                image_whites_lights_and_mids[i][j][2] = mid_threshold
                image_whites_lights_mids_and_darks[i][j][2] = dark_threshold
                value_study_image[i][j][2] = really_dark_value

                # Highlighted Pixels
                highlight_lights_image[i][j][1] = 255
                highlight_lights_image[i][j][2] = 255

                highlight_mids_image[i][j][1] = 255
                highlight_mids_image[i][j][2] = 255

                highlight_darks_image[i][j][1] = 255
                highlight_darks_image[i][j][2] = 255

                highlight_really_darks_image[i][j][1] = 255
                highlight_really_darks_image[i][j][2] = 255

    return image_whites_and_lights, image_whites_lights_and_mids, image_whites_lights_mids_and_darks, value_study_image, highlight_lights_image, highlight_mids_image, highlight_darks_image, highlight_really_darks_image


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

    sketch = produce_sketch(img)

    img, pil_palette = reduce_colours_and_produce_palette(img, 8)

    palette = numpy.asarray(pil_palette)

    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    value_study_list_HSV = list(retrieve_value_study(img))

    value_study_lights = cv.cvtColor(value_study_list_HSV[0], cv.COLOR_HSV2BGR)
    value_study_mids = cv.cvtColor(value_study_list_HSV[1], cv.COLOR_HSV2BGR)
    value_study_darks = cv.cvtColor(value_study_list_HSV[2], cv.COLOR_HSV2BGR)
    value_study_list_full_all_tones = cv.cvtColor(value_study_list_HSV[3], cv.COLOR_HSV2BGR)
    value_study_black_and_white = cv.cvtColor(value_study_list_full_all_tones, cv.COLOR_BGR2GRAY)

    highlight_lights_image = cv.cvtColor(value_study_list_HSV[4], cv.COLOR_HSV2BGR)
    highlight_mids_image = cv.cvtColor(value_study_list_HSV[5], cv.COLOR_HSV2BGR)
    highlight_darks_image = cv.cvtColor(value_study_list_HSV[6], cv.COLOR_HSV2BGR)
    highlight_really_darks_image = cv.cvtColor(value_study_list_HSV[7], cv.COLOR_HSV2BGR)

    value_study_list = [palette, sketch, value_study_black_and_white, value_study_lights, value_study_mids,
                        value_study_darks,
                        value_study_list_full_all_tones, highlight_lights_image, highlight_mids_image,
                        highlight_darks_image, highlight_really_darks_image]

    return value_study_list


def save_processed_image(image_list, filename):
    filename_base = os.path.basename(filename)[:-4]

    storage_location = 'resources\\produced_images\\' + '\\' + filename_base

    if not (os.path.exists(storage_location)):
        os.makedirs(storage_location)

    palette = image_list[0]
    sketch = image_list[1]
    value_study_black_and_white = image_list[2]
    value_study_lights = image_list[3]
    value_study_mids = image_list[4]
    value_study_darks = image_list[5]
    value_study_all_tones = image_list[6]
    highlight_lights_image = value_study_lights[7]
    highlight_mids_image = image_list[8]
    highlight_darks_image = image_list[9]
    highlight_really_darks_image = image_list[10]

    cv.imwrite(storage_location + '\\' + 'palette.png', palette)
    cv.imwrite(storage_location + '\\' + 'sketch.png', sketch)
    cv.imwrite(storage_location + '\\' + 'light_value_study.png', value_study_lights)
    cv.imwrite(storage_location + '\\' + 'mid_value_study.png', value_study_mids)
    cv.imwrite(storage_location + '\\' + 'dark_value_study.png', value_study_darks)
    cv.imwrite(storage_location + '\\' + 'full_value_study.png', value_study_all_tones)
    cv.imwrite(storage_location + '\\' + 'black_white_value_study.png',
               value_study_black_and_white)

    cv.imwrite(storage_location + '\\' + 'light_value_study_highlight.png', highlight_lights_image)
    cv.imwrite(storage_location + '\\' + 'mid_value_study_highlight.png', highlight_mids_image)
    cv.imwrite(storage_location + '\\' + 'dark_value_study_highlight.png', highlight_darks_image)
    cv.imwrite(storage_location + '\\' + 'really_dark_value_study_highlight.png', highlight_really_darks_image)


def create_ui():
    gui.theme("SystemDefault1")
    main_menu_layout = [
        [gui.Push()],
        [gui.Image("resources\\ui_images\\LogoTemp.png", key="-LOGO-")],
        [gui.Push(), gui.Button("Start Painting", key="-START-"), gui.Push()],
        [gui.Push(), gui.Button("Tutorial", key="-TUTORIAL-"), gui.Push()],
        [gui.Push()],
    ]

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
        [gui.Text("Choose an image to paint from left:")],
        [gui.Text(size=(40, 1), key="-IMAGE_TITLE-")],
        [gui.Image(key="-IMAGE-")],
        [gui.Button("Process This Photo?", key="-PROCESS-", visible=False)]
    ]

    painting_layout = [
        [gui.Push(), gui.Text("Current Step", key="-STEP_TRACKER-"), gui.Push()],
        [gui.VPush()],
        [gui.Image(key="-CURRENT_STEP-")],
        [gui.VPush()],
        [gui.Button("Previous Step", key="-PREVIOUS-"),
         gui.Push(),
         gui.Button("More Explanation", key="-EXPLAIN-"),
         gui.Push(),
         gui.Button("Next Step", tooltip="Next Step", key="-NEXT-")],
        [gui.Button("Save Image", key="-SAVE-"),
         gui.Push(),
         gui.Button("Highlight", key="-HIGHLIGHT-")],
    ]

    image_selection_layout = [
        [
            gui.Column(file_selector),
            gui.VSeperator(),
            gui.Column(image_to_process_column, key="-IMAGE_TO_PROCESS-"),
        ]
    ]

    final_layout = [
        [
            gui.Column(main_menu_layout, key="-MAIN_MENU-", visible=True, pad=(0, 20), justification='c'),
            gui.Column(image_selection_layout, key="-IMAGE_SELECTION-", visible=False, justification='c'),
            gui.Column(painting_layout, key="-PAINTING-", visible=False, justification='c'),
        ]
    ]

    display_window = gui.Window("Tech-Nicolour", final_layout, resizable=True, finalize=True)
    display_window.maximize()

    images = []
    current_picture = 0
    is_highlight: bool = False

    TOOLTIPS = ["This is your palette. This is the general collection of colours present in your final image.",
                "This is your sketch. This is will let you keep in mind the general outlines of objects while painting, which can be a bit hard to do.",
                "This is your value study. Use this to keep in mind the the eventual dark and bright spots of your final image - This will let you know what needs more paint!",
                "This is your lights layer of painting.",
                "This is your mids layer",
                "This is your darks layer",
                "This is your full image",
                "This is your lights highlight",
                "THis is your mids highlight",
                "This is your darks highlight",
                "This is your full image highlights"]

    STEPS = ["Palette", "Sketch", "Value Study", "Lights", "Mids", "Darks", "Final Image", "Lights Highlighted Changes",
             "Mids Highlighted Changes", "Darks Highlighted Changes", "Full Image Highlighted Changes"]

    STEP_EXPLANATIONS = [
        "Palette - This makes up the colours that will be present in your final painting.\n Get a feel for it now!",
        "Sketch - This will let you keep in mind the general outlines and shapes on objects as you're painting them.",
        "Value Study -",
        "Lights Layer -",
        "Mids Layer -",
        "Darks Layer -",
        "Final Layer -",
        "Lights Highlight - ",
        "Mids Highlight -",
        "Darks Highlight -",
        "Final Layer Highlight -"]

    while True:  # Event Loop
        event, values = display_window.read()

        if event == "Exit" or event == gui.WIN_CLOSED:
            break

        # Main Menu Page
        if event == "-START-":
            display_window["-MAIN_MENU-"].update(visible=False)
            display_window["-IMAGE_SELECTION-"].update(visible=True)

        # Image Selection
        elif event == "-FOLDER-":
            folder = values["-FOLDER-"]
            file_list = os.listdir(folder)

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

            display_window["-IMAGE_TITLE-"].update(filename)
            display_window["-IMAGE-"].update(data=image)
            display_window["-PROCESS-"].update(visible=True)

        elif event == "-PROCESS-":
            current_picture = 0
            display_window["-IMAGE_SELECTION-"].update(visible=False)
            images.clear()
            img = cv.imread(filename)

            executor = concurrent.futures.ThreadPoolExecutor()
            t = executor.submit(process_image, img, 1280)

            # while True:
            #     if t.running():
            #         gui.popup_animated(gui.DEFAULT_BASE64_LOADING_GIF,
            #                            message='Can Take Some Time \n Fun Fact:',
            #                            no_titlebar=True, keep_on_top=True, time_between_frames=100, text_color='black',
            #                            background_color='white')
            #     else:
            #         gui.popup_animated(None)
            #         break

            while t.running():
                gui.popup_animated(gui.DEFAULT_BASE64_LOADING_GIF,
                                       message='Can Take Some Time \n Fun Fact:',
                                       no_titlebar=True, keep_on_top=True, time_between_frames=100, text_color='black',
                                       background_color='white')


            gui.popup_animated(None)

            value_study_list = t.result()

            for image in value_study_list:
                temp_im = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                temp_im = Image.fromarray(temp_im)
                temp_im = ImageTk.PhotoImage(temp_im)
                images.append(temp_im)

            display_window["-PAINTING-"].update(visible=True)
            display_window["-NEXT-"].update(disabled=False)
            display_window["-PREVIOUS-"].update(disabled=True)
            display_window["-HIGHLIGHT-"].update(disabled=True)

            display_window["-CURRENT_STEP-"].update(data=images[current_picture])
            display_window["-CURRENT_STEP-"].set_tooltip(TOOLTIPS[current_picture])
            display_window["-STEP_TRACKER-"].update(STEPS[current_picture])

        # Painting Section
        elif event == "-NEXT-":
            current_picture = current_picture + 1
            display_window["-CURRENT_STEP-"].update(data=images[current_picture])
            display_window["-CURRENT_STEP-"].set_tooltip(TOOLTIPS[current_picture])
            display_window["-STEP_TRACKER-"].update(STEPS[current_picture])

            display_window["-PREVIOUS-"].update(disabled=False)
            if current_picture >= len(images) - 5:  # Stops the user from going into the highlighted images
                display_window["-NEXT-"].update(disabled=True)
            if current_picture >= 3:
                display_window["-HIGHLIGHT-"].update(disabled=False)

        elif event == "-PREVIOUS-":
            current_picture = current_picture - 1

            display_window["-CURRENT_STEP-"].update(data=images[current_picture])
            display_window["-CURRENT_STEP-"].set_tooltip(TOOLTIPS[current_picture])
            display_window["-STEP_TRACKER-"].update(STEPS[current_picture])

            display_window["-NEXT-"].update(disabled=False)
            if current_picture <= 0:
                display_window["-PREVIOUS-"].update(disabled=True)
            if current_picture < 3:
                display_window["-HIGHLIGHT-"].update(disabled=True)

        elif event == "-HIGHLIGHT-":
            if is_highlight:
                is_highlight = False
                display_window["-NEXT-"].update(disabled=False)
                display_window["-PREVIOUS-"].update(disabled=False)
                current_picture = current_picture - 4

                display_window["-CURRENT_STEP-"].update(data=images[current_picture])
                display_window["-CURRENT_STEP-"].set_tooltip(TOOLTIPS[current_picture])
                display_window["-STEP_TRACKER-"].update(STEPS[current_picture])

                display_window["-HIGHLIGHT-"].update("Highlight")
            elif not is_highlight:
                is_highlight = True
                display_window["-NEXT-"].update(disabled=True)
                display_window["-PREVIOUS-"].update(disabled=True)
                current_picture = current_picture + 4

                display_window["-CURRENT_STEP-"].update(data=images[current_picture])
                display_window["-CURRENT_STEP-"].set_tooltip(TOOLTIPS[current_picture])
                display_window["-STEP_TRACKER-"].update(STEPS[current_picture])

                display_window["-HIGHLIGHT-"].update("Revert")

            if current_picture <= 0:
                display_window["-PREVIOUS-"].update(disabled=True)
            if current_picture >= len(images) - 5:  # Stops the user from going into the highlighted images
                display_window["-NEXT-"].update(disabled=True)

        elif event == "-SAVE-":
            save_processed_image(value_study_list, filename)
            display_window["-IMAGE_SELECTION-"].update(visible=True)
            display_window["-PAINTING-"].update(visible=False)

        elif event == "-EXPLAIN-":
            gui.popup_ok(STEP_EXPLANATIONS[current_picture])

    display_window.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    create_ui()
