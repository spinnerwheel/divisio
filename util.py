import math
import os
import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage import io, morphology,util
from skimage.color import rgb2gray
from skimage.exposure import equalize_hist
from skimage.filters.rank import otsu
from skimage.io import imread, imread_collection
from skimage.util import img_as_ubyte
from sklearn.cluster import KMeans
from PIL import Image,ImageFilter

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
    hsv_image = image.convert('HSV')
    h_channel, s_channel, v_channel = hsv_image.split()
    return [np.array(h_channel), np.array(s_channel), np.array(v_channel)]

def ycbcr_filter(image):
    ycbcr_image = skimage.color.rgb2ycbcr(image)
    y_channel, cb_channel, cr_channel = ycbcr_image[:,:,0], ycbcr_image[:,:,1], ycbcr_image[:,:,2]
    return [y_channel, cb_channel, cr_channel]

def gaussian_filter(image, sigma):
    """
    Apply the gaussian filter to `image`.
    `image`: a PIL Image instance
    `sigma`: standard deviation of the gaussian kernel
    """
    filtered_image = skimage.filters.gaussian(image, sigma=sigma)
    return filtered_image

def connected_components(image):
    """
    Apply the connected components algorithm to `image`.
    `image`: an np.ndarray image
    """
    return skimage.measure.label(image, background=0)

def fill_holes(image):
    filled = util.invert(util.img_as_bool(image))
    filled = util.invert(util.img_as_float(morphology.binary_fill_holes(filled)))
    return filled


def alpha_trimmed(image, kernel=7):
    """
    Apply the alpha trimmed filter to `image`.
    `image`: an np.ndarray image
    `kernel`: dimension of the kernel, default=7
    """
    if kernel == 0:
        return image
    tmp = np.copy(image)
    for x in range(0, len(tmp)-kernel, kernel):
        for y in range(0, len(tmp[0])-kernel, kernel):
            textel = tmp[x:x+kernel, y:y+kernel]
            textel = np.sort(textel, axis=None)
            textel = textel[1:-1]
            tmp[x:x+kernel, y:y+kernel] = np.mean(textel)
    return Image.fromarray(tmp)

def canny_filter(image, sigma,lb,ub):
    return skimage.feature.canny(image, sigma=sigma, low_threshold=lb, high_threshold=ub)

def erode_image(image,kernel):
    return skimage.morphology.erosion(image, kernel)

def bilateral_filter(image):
    return skimage.restoration.denoise_bilateral(image,channel_axis=-1)

def dilate_image(image,kernel):
    return skimage.morphology.dilation(image, kernel)

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
        image = skimage.io.imread(os.path.join(folder,filename))
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
