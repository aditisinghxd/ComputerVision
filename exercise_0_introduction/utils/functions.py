import numpy as np
from typing import List, Tuple
import cv2
import os

t_image_list = List[np.array]
t_str_list = List[str]
t_image_triplet = Tuple[np.array, np.array, np.array]


def show_images(images: t_image_list, names: t_str_list) -> None:
    for img, name in zip(images, names):
        cv2.imshow(name, img) # Display the image in a window
    cv2.waitKey(0) # Wait until a key is pressed
    cv2.destroyAllWindows() # Close all windows


def save_images(images: t_image_list, filenames: t_str_list, **kwargs) -> None:
    for img, filename in zip(images, filenames):
        cv2.imwrite(filename,img) #save the image



def scale_down(image: np.array) -> np.array:
    height, width = image.shape[:2]
    return cv2. resize(image, (width//2, height//2)) # Resize image to half


def separate_channels(colored_image: np.array) -> t_image_triplet:
    blue_channel = np.zeros_like(colored_image)
    green_channel = np.zeros_like(colored_image)
    red_channel = np.zeros_like(colored_image)

    blue_channel[:, :, 0] = colored_image[:, :, 0]  # Blue channel
    green_channel[:, :, 1] = colored_image[:, :, 1]  # Green channel
    red_channel[:, :, 2] = colored_image[:, :, 2]    # Red channel
    
    return blue_channel, green_channel, red_channel
