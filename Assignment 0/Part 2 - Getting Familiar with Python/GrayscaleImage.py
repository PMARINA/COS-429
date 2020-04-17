import numpy as np
import cv2
import os
window_title = "The Input Image"
input_image = "input.jpg"
output_image = os.path.basename(__file__)[:-len(".py")] + ".jpg"

def read_image(file_name = input_image):
    img = cv2.imread(file_name)
    return img

def display_image(img,window_title = window_title):
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.imshow(window_title,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

def grayscale(img):
    grayscale = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #=6, BGR and not RGB because of how cv2 returns images
    return grayscale

def save_to_disk(img,filename=output_image):
    cv2.imwrite(filename,img)

if __name__ == "__main__":
    img = grayscale(read_image())
    save_to_disk(img)
    display_image(img)
