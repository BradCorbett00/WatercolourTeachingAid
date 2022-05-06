import cv2 as cv
import numpy
import numpy as np
import os.path
from PIL import Image
from PIL import ImageDraw


def produce_sketch(image):
    grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    inverted = cv.bitwise_not(grey)

    blurred = cv.GaussianBlur(inverted, (51, 51), 0)

    inverted_blur = cv.bitwise_not(blurred)

    final_sketch = cv.divide(grey, inverted_blur, scale=256.0)

    return final_sketch


def reduce_colours_and_produce_palette(image, k):
    """
    Utilises k-means clustering to reduce the amount of colours in an image to make it appear more watercolour-esque
    :param image: The image that will have its images reduced.
    :param k: The number of images it will be reduced to
    :return final_img: The image after it has had its number of colours reduced
    :return palette: A drawing made up of each colour in the image.
    """
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
    """
    Thresholds the image at different levels to identify the whites, lights, mids, darks, and really darks, as well as generating the images to highlight the differences
    :param image: The image to be thresholded.
    :return: A list of different images, as well as the highlighted images.
    """
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
    """

    :param img: The image that will shrink.
    :param target_dimension: The number that the dimensions should be reduced to, or under.
    :return: The new size of the image
    """
    width = int(img.shape[1])
    height = int(img.shape[0])

    new_size = reduce_resolution(height, width, target_dimension)

    new_img = cv.resize(img, new_size, interpolation=cv.INTER_AREA)

    return new_img


def resize_pil_image(image_to_shrink, target_dimension) -> tuple[int, int]: # Works with a PIL image
    """
    Shrinks a Pillow image
    :param image_to_shrink: Self explanatory
    :param target_dimension: The number that the dimensions should be reduced to, or under.
    :return: The new size of the image.

    """
    width = int(image_to_shrink.size[0])
    height = int(image_to_shrink.size[1])

    new_size = reduce_resolution(height, width, target_dimension)

    return new_size


def reduce_resolution(height, width, target_dimension):
    """

    :param height: The initial height
    :param width: The initial width
    :param target_dimension: The number that both dimensions should be reduced under.
    :return: Returns a tuple of the new size of the image.
    """
    image_below_target: bool = ((width <= target_dimension) and (height <= target_dimension))
    while not image_below_target:
        width = int(width * 0.75)
        height = int(height * 0.75)
        image_below_target = ((width <= target_dimension) and (height <= target_dimension))
    new_size = (width, height)
    return new_size


def process_image(img, target_dimension):
    """
    Produces a variety of images that will be later displayed onto the UI

    :param img: The image to be processed
    :param target_dimension: The resolution that the image will be reduced to save performance time
    :return: Returns a list of each of the outputs made by processing the image
    """
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
    """
    Will create a folder and save each of the outputs of a processed image into said folder.

    :param image_list: The list of images which will be saved
    :param filename: The filename of the original image
    :return: Nothing
    """
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