import concurrent.futures
import PySimpleGUI as gui
import os.path
from PIL import ImageTk
import random

from image_manip_file import *


def create_ui():
    gui.theme("SystemDefault1")
    main_menu_layout = [
        [gui.Push()],
        [gui.Image("resources\\ui_images\\Logo.png", key="-LOGO-")],
        [gui.Push(), gui.Button("Start Painting", key="-START-"), gui.Push()],
        [gui.Push()],
        [gui.Push(), gui.Text('"https://www.freepik.com/vectors/background" Background vector created by rawpixel.com - www.freepik.com'), gui.Push()]
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
                "This is your lights layer of painting. This is going to be your first step of painting.",
                "This is your mids layer. Build up on what you can tell is a little different.",
                "This is your darks layer. Same as before, go a little darker here.",
                "This is your full image. Add the finishing touches of darkening here.",
                "This is your lights highlight",
                "THis is your mids highlight",
                "This is your darks highlight",
                "This is your full image highlights"]

    STEPS = ["Palette", "Sketch", "Value Study", "Lights", "Mids", "Darks", "Final Image", "Lights Highlighted Changes",
             "Mids Highlighted Changes", "Darks Highlighted Changes", "Full Image Highlighted Changes"]

    STEP_EXPLANATIONS = [
        "Palette - This makes up the colours that will be present in your final painting.\n Get a feel for it now!",
        "Sketch - This will let you keep in mind the general outlines and shapes on objects as you're painting them.",
        "Value Study - This lets you identify the light and dark areas of the image, and also the composition.",
        "Lights Layer - This is your first layer of paints. For now, you only need to worry about what's white, and what's not.",
        "Mids Layer - Build up on the darker areas from before. If you can't tell, click the Highlight button.",
        "Darks Layer - Same as before, keep building up the now even darker areas.",
        "Final Layer - One last layer of paints to apply here. Add the finishing touches, and really emphasise the deepness of the dark colours here.",
        "Lights Highlight - The bright coloured parts are what you want to paint in this step. Try turning Highlight off and on if you can't tell.",
        "Mids Highlight - The bright coloured parts are what you want to paint in this step. Try turning Highlight off and on if you can't tell.",
        "Darks Highlight - The bright coloured parts are what you want to paint in this step. Try turning Highlight off and on if you can't tell.",
        "Final Layer Highlight - The bright coloured parts are what you want to paint in this step. Try turning Highlight off and on if you can't tell."]

    PAINTING_TIPS = [
        "Don't forget to keep a clean brush!",
        "Try out painting with a little bit less water on the brush sometimes!",
        "Keep paper towels on hand!",
        "Keep a variety of brushes on hand, smaller ones can be great for details!",
        "Take your time, make sure to have fun!"
    ]

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
            loading_tip = PAINTING_TIPS[random.randint(0, len(PAINTING_TIPS) - 1)]
            while t.running():
                gui.popup_animated(gui.DEFAULT_BASE64_LOADING_GIF,
                                       message='Can Take Some Time \n ' + loading_tip,
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
            display_window["-STEP_TRACKER-"].update(STEPS[current_picture] + " - Hover over for a brief explanation, or click 'More Explanation' for in-depth.")

        # Painting Section
        elif event == "-NEXT-":
            current_picture = current_picture + 1
            display_window["-CURRENT_STEP-"].update(data=images[current_picture])
            display_window["-CURRENT_STEP-"].set_tooltip(TOOLTIPS[current_picture])
            display_window["-STEP_TRACKER-"].update(STEPS[current_picture] + " - Hover over for a brief explanation, or click 'More Explanation' for in-depth.")

            display_window["-PREVIOUS-"].update(disabled=False)
            if current_picture >= len(images) - 5:  # Stops the user from going into the highlighted images
                display_window["-NEXT-"].update(disabled=True)
            if current_picture >= 3:
                display_window["-HIGHLIGHT-"].update(disabled=False)

        elif event == "-PREVIOUS-":
            current_picture = current_picture - 1

            display_window["-CURRENT_STEP-"].update(data=images[current_picture])
            display_window["-CURRENT_STEP-"].set_tooltip(TOOLTIPS[current_picture])
            display_window["-STEP_TRACKER-"].update(STEPS[current_picture] + " - Hover over for a brief explanation, or click 'More Explanation' for in-depth.")

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

