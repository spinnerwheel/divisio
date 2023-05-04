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

t_size = 9
t_step = math.floor(t_size-2)
functions = [
    get_mean,
    get_stdev,
    #get_LBP
]
result = []
titles = []
path = './benchmark_images/'
images_title = os.listdir(path)
for title in images_title:
    text,obj,num = title.split('_')
    titles.append(text+'_'+obj)
    
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,5))

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

def sobel(image):
    sobel_x = (cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=9))
    sobel_y = (cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=9))
    image = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    return image

if __name__ == '__main__':
    image_list = load_images_from_folder(path)
    for image in image_list:
        print(f"Processing image {len(result)+1}/{len(image_list)}...", end="\r")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = alpha_trimmed(image,7)       
        image = sobel(image)
                
        #image = np.concatenate((image_alfa, image_median), axis=1)
    # applica un filtro di Sobel per individuare i bordi dell'immagine
        #out, rows, cols = compute_local_descriptor(image, t_size, t_step, functions)
        #image = cluster(out, rows, cols)
        result.append(image)
        
    plot(result,titles)