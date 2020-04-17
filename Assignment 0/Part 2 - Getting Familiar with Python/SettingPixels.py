import numpy as np
import cv2
import os
window_title = "The Input Image"
input_image = "input.jpg"
output_image = os.path.basename(__file__)[:-len(".py")] + ".jpg"
HORIZONTAL = 0
VERTICAL = 1

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

def get_dimensions_hw(img):
    return img.shape[0:2]

def get_middle_pixels_hw(img, new_height, new_width):
    input_img_h,input_img_w = get_dimensions_hw(img)
    if new_height > input_img_h:
        raise ValueError("Requested new height (" + str(new_height) + ") is greater than image height (" + str(input_img_h) + ").")
    if new_width > input_img_w:
        raise ValueError("Requested new width (" + str(new_width) + ") is greater than image width (" + str(input_img_w) + ").")
    middle_h = round(input_img_h/2)
    half_new_height = round(new_height/2)
    middle_w = round(input_img_w/2)
    half_new_width = round(new_width/2)
    middle_pixels = img[middle_h-half_new_height:middle_h+half_new_height,middle_w-half_new_width:middle_w+half_new_width]
    return middle_pixels

def set_periodic_pixel(img, frequency, direction, new_pixel):
    h,w = get_dimensions_hw(img)
    img = np.array(img,copy=True)
    if direction == HORIZONTAL:
        for i in range(0,h):
            for j in range(0,w,frequency):
                img[i][j] = new_pixel
    elif direction == VERTICAL:
        for i in range(0,h,frequency):
            for j in range(0,w):
                img[i][j] = new_pixel
    return img   
            

if __name__ == "__main__":
    img = read_image()
    revised = set_periodic_pixel(img,10,HORIZONTAL,0)
    revised = set_periodic_pixel(revised, 20, VERTICAL, 0)
    save_to_disk(revised)
    display_image(revised)

#Note: Owing to the large input image used for this example, the program will not show all
#lines unless you zoom in on the saved file (unless your monitor happens to have enough
#resolution...)
