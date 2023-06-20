import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

# pylint: disable = no-name-in-module

# Constants
PATH = ""
WELCOME ="""
 ____ ____ ____ ____ ____ ____ ____ 
||d |||i |||v |||i |||s |||i |||o ||
||__|||__|||__|||__|||__|||__|||__||
|/__\|/__\|/__\|/__\|/__\|/__\|/__\|


"""

def load_images_from_folder(folder, return_filenames=False):
    """
    Load images from `folder` and return RGB version of it
    Can optionally returns filenames of those images
    """
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
        if len(images) == 1:
            plt.imshow(images[0], cmap="gray")
            if suptitle is not None:
                plt.title(suptitle)
        else:
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

def median_filter(image, kernel_size):
    """Returns the median filtered image"""
    return cv2.medianBlur(image, kernel_size)

def ycbcr_filter(image):
    """Returns the YCbCr image and the Y channel"""
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    y_channel = ycbcr_image[:,:,0]
    return ycbcr_image, y_channel

def gray_filter(image):
    """Returns the gray image"""
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def gaussian_filter(image, sigma):
    """Returns the gaussian filtered image of `sigma`"""
    return cv2.GaussianBlur(image, (0,0), sigma)

def canny_filter(image, sigma, low_threshold, high_threshold):
    """Returns the canny filtered image"""
    return cv2.Canny(image, low_threshold, high_threshold, sigma)

def dilate_image(image, kernel):
    """Returns the dilated image using `kernel`"""
    return cv2.dilate(image, kernel)

def erode_image(image, kernel):
    """Returns the eroded image using `kernel`"""
    return cv2.erode(image, kernel)

def bilateral_filter(image, sigma_color, sigma_space):
    """Returns the bilateral filtered image
    image = image to be filtered
    sigma_color = filter sigma in the color space (the higher the more colors are considered)
    sigma_space = filter sigma in the coordinate space  (the higher the more pixels are considered)
    """
    return cv2.bilateralFilter(image)

def contornus(image,treshold):
    """Returns the contornus of the image"""
    # find contours in the binary image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # create a new blank image
    # iterate over all the contours and draw them onto the new image
    for i, contour in enumerate(contours):
        if len(contour) > treshold:
            new_img = np.zeros(image.shape, np.uint8)
            cv2.drawContours(new_img, [contour], 0, (255, 255, 255), -1)
            cv2.imwrite(f'./splitted/output_image{i}.png', new_img)
    # save the new image
    
def morphology_fill(image):
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(image, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    filled = cv2.subtract(dilation, image)

def connected_components(image):
    """Returns the connected components of the image"""
    return cv2.connectedComponents(image)


class circleGrowing:
    image = None
    output_image = None
    inside = False
    onEdge = False
    ended = False
    results = []

    def __init__(self, image=None):
        if image is None:
            self.image = self._generate_mock_image(64, 64)
        else:
            self.image = image
        self.colums = self.image.shape[0]
        self.rows = self.image.shape[1]
        self.output_image = self.image.copy() #np.zeros(self.image.shape).astype(np.uint8)
        self.marked = np.stack([self.image]*3, axis=-1)


    def _reset(self):
        self.inside = False
        self.onEdge = False


    def _mark(self, row, col):
        self.output_image[row, col] = 255
        self.marked[row, col] = [255, 0, 0]


    def _generate_mock_image(self, width, height):
        image = np.zeros((width, height))
        center_x = width // 2
        center_y = height // 2
        radius = [20, 15, 10, 5]
        xx, yy = np.mgrid[:height, :width]
        distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        image[distance < radius[0]] = 255
        image[distance < radius[1]] = 0
        image[distance < radius[2]] = 255
        image[distance < radius[3]] = 0
        return image


    def _next_neighbors_edge(self, neighbors):
        """The next internal edge should be one that has value 255 and has a 0 near him"""
        indices = np.argwhere(neighbors == 255)
        for i, j in indices:
            if (i > 0 and neighbors[i-1, j] == 0):
                return i-1,j
            if (i < 2 and neighbors[i+1, j] == 0):
                return i+1,j
            if (j > 0 and neighbors[i, j-1] == 0):
                return i, j-1
            if (j < 2 and neighbors[i, j+1] == 0):
                return i, j+1
        return None, None


    def _get_8_neighbors(self, row, col, image=None):
        if image is None:
            image = self.image
        return image[row-1:row+2, col-1:col+2]


    def _next_internal_edge(self, row, col):
        self.image[row, col] = 128
        neighbors = self._get_8_neighbors(row, col)
        n, m = self._next_neighbors_edge(neighbors)
        if n is None or m is None:
            return n, m
        return row+n-1, col+m-1


    def _first_internal_edge(self):
        index = 0
        row, col = (None, None)bibite
        while not self.inside:
            if self.onEdge is True:
                if self.image[index+1, index+1] == 0:
                    self.inside = True
                    row, col = (index, index)
            if self.image[index, index] == 255:
                self.onEdge = True
            index+=1
            # failsafe condition, to break a possible loop
            if index == 10000:
                choise = input("10000 iterations passed. Want to continue? [y/N]")
                if choise.upper() != "Y":
                    break
        return row, col


    def _recursive_layer_call(self, row, col):
        if row is None or col is None:
            pass
        else:
            self._mark(row, col)
            row, col = self._next_internal_edge(row, col)
            self._recursive_layer_call(row, col)


    def _recursive_image_call(self):
        for i in range(10):
            self._reset()
            row, col = self._first_internal_edge()
            self._mark(row, col)
            self._recursive_layer_call(row, col)
            self.image = self.output_image.copy()
            plt.imshow(self.image)
            plt.show()

    def start(self):
        row, col = self._first_internal_edge()
        # print(f"Point: {row}, {col}\t", end="\r")
        self._mark(row, col)
        self._recursive_image_call()
