import numpy as np
import cv2
window_title = "The Input Image"
input_image = "input.jpg"

def read_image(file_name = input_image):
    img = cv2.imread(file_name)
    return img

def display_image(img,window_title = window_title):
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.imshow(window_title,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

if __name__ == "__main__":
    display_image(read_image())
