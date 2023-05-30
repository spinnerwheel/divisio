import cv2
import os
import math
import numpy as np
import matplotlib.pyplot as plt

# pylint: disable = no-name-in-module

# Constants
PATH = "/path/to/divisio/folder"
WELCOME ="""
 ____ ____ ____ ____ ____ ____ ____ 
||d |||i |||v |||i |||s |||i |||o ||
||__|||__|||__|||__|||__|||__|||__||
|/__\|/__\|/__\|/__\|/__\|/__\|/__\|


"""

def load_images_from_folder(folder, return_filenames=False):
    images = []
    filenames = os.listdir(folder)
    for filename in filenames:
        image = cv2.imread(os.path.join(folder,filename))
        images.append(image)
    if return_filenames is True:
        return images, filenames
    return images

def plot(images, titles=None, suptitle=None):
    """
    Crea un plot della libreria matplotlib e mostra le immagini 
    con corrispettivo titolo
    """
    if isinstance(images, np.ndarray):
        plt.imshow(images, cmap="gray")
        if suptitle is not None:
            plt.title(suptitle)
    elif isinstance(images, list):
        dim = math.ceil(math.sqrt(len(images)))
        figure, axes = plt.subplots(dim, dim, sharex=True, sharey=True, constrained_layout=True)
        axs = axes.ravel()
        if suptitle is not None and isinstance(suptitle, str):
            figure.suptitle(suptitle)
        if titles is None:
            for a, i in zip(axs, images):
                a.imshow(i)
        else:
            for a, i, t in zip(axs, images, titles):
                a.imshow(i)
                a.set_title(t)
    else:
        print("Il tipo degli oggetti passati non è supportato.")
        print(f"images: {type(images)} (should be a list of images or an image)")
        if titles is not None:
            print(f"tiles: {type(titles)} (should be a list of str or a str)")
        if suptitle is not None:
            print(f"suptitle: {type(suptitle)} (should be str)")
    plt.show()


def save(images, filenames, dir="saved"):
    """
    Saves the given `images` in the directory `dir` as `filenames`
    """
    path = os.path.join(PATH, dir)
    if not os.path.isdir(path):
        os.mkdir(path)
    if isinstance(images, np.ndarray) and isinstance(filenames, str):
        fname = os.path.join(path, filenames+".jpg")
        plt.imsave(fname, images)
    elif isinstance(images, list) and isinstance(filenames, list):
        for img, title in zip(images, filenames):
            save(img, title)
    else:
        print("Il tipo degli oggetti passati non è supportato.")
        print(f"images: {type(images)} (should be a list of images or an image)")
        print(f"filenames: {type(filenames)} (should be a list of str or a str)")

#write a script that returns the y channel of the ycbcr image
def ycbcr_filter(image):
    """Returns the YCbCr image and the Y channel"""
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    y_channel = ycbcr_image[:,:,0]
    return ycbcr_image, y_channel

def gray_filter(image):
    """Returns the gray image"""
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

#write a function that takes in input an image and a sigma value and returns the gaussian filtered image
def gaussian_filter(image, sigma):
    """Returns the gaussian filtered image"""
    return cv2.GaussianBlur(image, (0,0), sigma)

#write a function that takes in input an image and a sigma value and returns the canny filtered image
def canny_filter(image, sigma, low_threshold, high_threshold):
    """Returns the canny filtered image"""
    return cv2.Canny(image, low_threshold, high_threshold, sigma)

#write a function that takes in input an image and a kernel and returns the dilated image
def dilate_image(image, kernel):
    """Returns the dilated image"""
    return cv2.dilate(image, kernel)

#write a function that takes in input an image and a kernel and returns the eroded image
def erode_image(image, kernel):
    """Returns the eroded image"""
    return cv2.erode(image, kernel)

def bilateral_filter(image, sigma_color, sigma_space):
    """Returns the bilateral filtered image
    image = image to be filtered
    sigma_color = filter sigma in the color space (the higher the more colors are considered)
    sigma_space = filter sigma in the coordinate space  (the higher the more pixels are considered)
    """
    return cv2.bilateralFilter(image)

def contornus(image):
    """Returns the contornus of the image"""
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_fill = np.zeros_like(image)
    cv2.drawContours(img_fill, contours, -1, 255, cv2.FILLED)
    cv2.imshow("contornus", img_fill)
    cv2.waitKey(0)
    i = 0
    
def morphology_fill(image):
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(image, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    filled = cv2.subtract(dilation, image)

def connected_components(image):
    """Returns the connected components of the image"""
    return cv2.connectedComponents(image)