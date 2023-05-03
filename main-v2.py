import os
import csv
import math
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.filters import gaussian
from sklearn.cluster import KMeans
from skimage import data
import cv2
from compute import *
from PIL import Image
from script_utili import *

t_size = 7
t_step = math.floor(t_size/2)
functions = [
    get_mean,
    get_stdev
    #get_LBP
]

result = []
titles = ['legno','marmo','tessuto']
path = './new_photos/'

def cluster(out, rows, cols):
    label = KMeans(n_clusters=2, n_init="auto").fit(out).labels_
    return label.reshape(rows,cols)

def plot(images, titles):
    dim = math.ceil(math.sqrt(len(images)))
    figure, axes = plt.subplots(dim, dim, sharex=True, sharey=True) 
    axs = axes.ravel()
    for i in range(0, len(images)):
        axs[i].imshow(images[i])
        axs[i].set_title(titles[i])
    plt.show()

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

if __name__ == '__main__':
    #per ogni immagine nella cartella new_photos
    image_list = load_images_from_folder(path)
    for image in image_list:
        print(f"Processing image {len(result)+1}/{len(image_list)}...", end="\r")
        image = cv2.resize(image, (256,256))
        #image = histogram_equalization(image)           
        #image = gaussian_filter(image,)
        out, rows, cols = compute_local_descriptor(image, t_size, t_step, functions)
        image_out = cluster(out, rows, cols)
        result.append(image_out)
        
    plot(result, titles)