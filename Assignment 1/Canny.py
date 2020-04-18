import numpy as np
import cv2
import os
import math
import pylab as p
import numpy as np

window_title = "The Input Image"
input_image = "input.jpg"
output_image = os.path.basename(__file__)[:-len(".py")] + ".jpg"
HORIZONTAL = 0
VERTICAL = 1


def read_image(file_name=input_image):
    img = cv2.imread(file_name)
    return img


def display_image(img, window_title=window_title):
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.imshow(window_title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


def grayscale(img):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # =6, BGR and not RGB because of how cv2 returns images
    return grayscale


def save_to_disk(img, filename=output_image):
    cv2.imwrite(filename, img)


def get_dimensions_hw(img):
    return img.shape[0:2]


def get_middle_pixels_hw(img, new_height, new_width):
    input_img_h, input_img_w = get_dimensions_hw(img)
    if new_height > input_img_h:
        raise ValueError(
            "Requested new height (" + str(new_height) + ") is greater than image height (" + str(input_img_h) + ").")
    if new_width > input_img_w:
        raise ValueError(
            "Requested new width (" + str(new_width) + ") is greater than image width (" + str(input_img_w) + ").")
    middle_h = round(input_img_h / 2)
    half_new_height = round(new_height / 2)
    middle_w = round(input_img_w / 2)
    half_new_width = round(new_width / 2)
    middle_pixels = img[middle_h - half_new_height:middle_h + half_new_height,
                    middle_w - half_new_width:middle_w + half_new_width]
    return middle_pixels


def set_periodic_pixel(img, frequency, direction, new_pixel):
    h, w = get_dimensions_hw(img)
    img = np.array(img, copy=True)
    if direction == HORIZONTAL:
        for i in range(0, h):
            for j in range(0, w, frequency):
                img[i][j] = new_pixel
    elif direction == VERTICAL:
        for i in range(0, h, frequency):
            for j in range(0, w):
                img[i][j] = new_pixel
    return img


def flip(img, direction):
    h, w = get_dimensions_hw(img)
    flipped = np.array(img, copy=True)
    if direction == HORIZONTAL:
        for i in range(h):
            for j in range(w):
                flipped[i][j] = img[i][w - j - 1]
    elif direction == VERTICAL:
        for i in range(h):
            for j in range(w):
                flipped[i][j] = img[h - i - 1][j]
    return flipped


def show_side_by_side(img1, img2):
    h1, w1 = get_dimensions_hw(img1)
    h2, w2 = get_dimensions_hw(img2)
    side_by_side = np.zeros([max(h1, h2), w1 + w2, 3], np.uint8)
    for i in range(h1):
        for j in range(w1):
            side_by_side[i][j] = img1[i][j]
    for i in range(h2):
        for j in range(w2):
            side_by_side[i][j + w1] = img2[i][j]
    return side_by_side


def one_dimensional_gaussian(x, sigma=1):
    g = 1.0 / (pow(2 * math.pi, 0.5) * sigma) * pow(math.e, -x * x / (2 * sigma * sigma))
    return g


def diff_one_dimensional_gaussian(x, k, sigma=1):
    return (one_dimensional_gaussian(x + k) - one_dimensional_gaussian(x)) / (k)


def second_diff_one_dimensional_gaussian(x, k, sigma=1):
    return (diff_one_dimensional_gaussian(x + k, k) - diff_one_dimensional_gaussian(x, k)) / (k)


# Below functions are required for this assignment:
def filtered_gradient(im, sigma):
    f_x, f_y = None
    return f_x, f_y


def edge_strength_and_orientation(f_x, f_y):
    f, d = None
    return f, d


def suppression(f, d):
    i = None
    return i


def hysteresis_thresholding(i, d, t_l, t_h):
    edge_map = None
    return edge_map


def canny_edge_detection(im, sigma, t_l, t_h):
    edge_map = None
    return edge_map


if __name__ == "__main__":
    pass
