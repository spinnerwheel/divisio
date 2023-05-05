from scipy import ndimage
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from script_utili import *

def sobel(image):
    sobel_x = (cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=9))
    sobel_y = (cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=9))
    image = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    return image

def plot(images, titles):
    dim = math.ceil(math.sqrt(len(images)))
    figure, axes = plt.subplots(dim, dim, sharex=True, sharey=True) 
    axs = axes.ravel()
    for i in range(0, len(images)):
        axs[i].imshow(images[i], cmap="gray")
        axs[i].set_title(titles[i])
    plt.show()
    
if __name__ == '__main__':
    images = load_images_from_folder('./benchmark_images/')
    img = images[0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = ndimage.morphological_gradient(gray, size=(1,9))
    gray = cv2.Canny(gray, 100, 200)
    gray = ndimage.morphological_gradient(gray, size=(1,9))
    gray = 
    plt.imshow(gray, cmap='gray')
    plt.show()
