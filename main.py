import math

from compute import *
from util import *
import argparse


if __name__ == "__main__":
    # definizione delle variabili
    folder = "./benchmark_images"
    parser = argparse.ArgumentParser(description="Divisio")
    parser.add_argument("-aph", type=int, help="alpha trimmed kernel",default=7)
    parser.add_argument("-canny", type=float, help="canny sigma", default=2)
    parser.add_argument("--l",dest="low", type=float, required=False, help="canny low threshold", default=None)
    parser.add_argument("--h",dest="high", type=float, required=False, help="canny high threshold", default=None)
    
    args = parser.parse_args()

    print(WELCOME)

    images, filenames = load_images_from_folder(folder, return_filenames=True)
    result = []
    kernel = skimage.morphology.disk(7)
    for image in images[:4]:
        print(f"Processing image {len(result)}/{len(images)}...", end="\r")
        image = ycbcr_filter(image)[0]
        image = gaussian_filter(image, 2.2)
        image = canny_filter(image, 1.2,25,55)
        imge = fill_holes(image)
        result.append(image)
        
    plot(result, filenames)
