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
    dilate_kernel = skimage.morphology.square(8)
    erode_kernel = skimage.morphology.square(7)
    for image in images:
        print(f"Processing image {len(result)+1}/{len(images)}...", end="\r")
        #resize the image with cv2 to 128x128
        image = cv2.resize(image, (128,128))
        image = ycbcr_filter(image)[0]
        image = gaussian_filter(image, 2.4)
        image = canny_filter(image, 1,30,60)
        image = dilate_image(image, dilate_kernel)
        #image = contornus(image)
        result.append(image)
        
    plot(result, filenames)
