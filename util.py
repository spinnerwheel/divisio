import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.exposure import equalize_hist
from skimage.filters.rank import otsu
from skimage.io import imread, imread_collection
from skimage.util import img_as_ubyte
from sklearn.cluster import KMeans

# pylint: disable = no-name-in-module

# Constants
PATH = "/path/to/divisio/folder"
WELCOME ="""
 ____ ____ ____ ____ ____ ____ ____ 
||d |||i |||v |||i |||s |||i |||o ||
||__|||__|||__|||__|||__|||__|||__||
|/__\|/__\|/__\|/__\|/__\|/__\|/__\|


"""

def hsv_filter(image):
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return [hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]]

def ycbcr_filter(image):
    # Convert to YCrCb color space and separate the Y channel
    ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    return [ycbcr[:, :, 0], ycbcr[:, :, 1], ycbcr[:, :, 2]]

def binarize(image, footprint=None):
    """
    Binarization function using `otsu` method from skimage
    """
    # Threshold the image
    if footprint is None:
        # footprint diventa una copia dell'immagine con tutti a valori a 1
        footprint = np.ones([len(image), len(image)])
    binarized = otsu(image, footprint)
    return binarized

def alpha_trimmed(image, kernel=7):
    """
    Apply the alpha trimmed filter to `image`.
    `image`: an np.ndarray image
    `kernel`: dimension of the kernel, default=7
    """
    tmp = np.copy(image)
    for x in range(0, len(tmp)-kernel, kernel):
        for y in range(0, len(tmp[0])-kernel, kernel):
            textel = tmp[x:x+kernel, y:y+kernel]
            textel = np.sort(textel, axis=None)
            textel = textel[1:-1]
            tmp[x:x+kernel, y:y+kernel] = np.mean(textel)
    return tmp

def sobel(image):
    sobel_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=9)
    sobel_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=9)
    image = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    return image

def normalize(image):
    out = img_as_ubyte(image)
    return out

def cluster(out, rows, cols):
    label = KMeans(n_clusters=3, n_init="auto").fit(out).labels_
    return label.reshape(rows, cols)

# Utility relative al caricamento e mostra di immagini

def load_images_from_folder(folder, return_filenames=False):
    images = []
    filenames = os.listdir(folder)
    for filename in filenames:
        image = imread(os.path.join(folder, filename))
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
        figure, axes = plt.subplots(dim, dim, sharex=True, sharey=True)
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
