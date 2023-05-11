import math

from compute import *
from util import *

# definizione delle variabili
folder = "./benchmark_images"

print(WELCOME)

images, filenames = load_images_from_folder(folder, return_filenames=True)
result = []
for image in images[:4]:
    print(f"Processing image {len(result)}/{len(images)}...", end="\r")
    ###
    # the actual processing of the images
    ###
    result.append(image)
plot(result, filenames)
