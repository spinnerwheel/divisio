import os
import csv
import math
import matplotlib.pyplot as plt
from skimage.io import imread
from sklearn.cluster import KMeans
from skimage import data

from compute import *

welcome ="""
 ____ ____ ____ ____ ____ ____ ____ ____ ____ ____ ____ ____ ____ ____ 
||c |||l |||a |||s |||s |||i |||f |||i |||c |||a |||t |||o |||r |||e ||
||__|||__|||__|||__|||__|||__|||__|||__|||__|||__|||__|||__|||__|||__||
|/__\|/__\|/__\|/__\|/__\|/__\|/__\|/__\|/__\|/__\|/__\|/__\|/__\|/__\|


"""

images = []
with open("./out.csv", "r") as f:
    reader = csv.reader(f)
    images = [row for row in reader]

# definizione delle variabili
path = "./benchmark_images"
t_size = 11
t_step = math.floor(t_size/2)
functions = [
    get_stdev,
    get_mean,
    get_LBP
]

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
    plt.show(block=False)
    plt.waitforbuttonpress(timeout=30)
    plt.close("all")
    
print(welcome)

files = [os.path.join(path, "resized"+image[2]+".jpg") for image in images]

result = []
titles = [image[0]+" "+image[1] for image in images]
for f in files[:2]:
    print(f"Processing image {len(result)+1}/{len(files)}...", end="\r")
    img = imread(f)
    out, rows, cols = compute_local_descriptor(img, t_size, t_step, functions)
    image_out = cluster(out, rows, cols)
    result.append(image_out)

plot(result, titles)
print("\nDone. Bye bye")