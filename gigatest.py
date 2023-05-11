import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage.color import rgb2gray
from skimage.feature import canny

from util import *


# I want to test all possible and different parameters for Canny and see
# which one is the best for our purpose
def test_parameters():
    i = 1
    images = load_images_from_folder("benchmark_images/")
    for image in images:
        image = rgb2gray(image)
        image = skimage.util.img_as_ubyte(image)
        print(f"Processing image {i}/{len(images)}...", end="\r")
        # for sigma in np.linspace(1, 2.2, 7):
        sigma = 2.2
        for low_threshold in range(30, 50, 5):
            for high_threshold in range(80, 100, 5):
                img = canny(image, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
                suptitle = f"{i}-{sigma}-{low_threshold}-{high_threshold}"
                save(img, suptitle)
        i+=1
