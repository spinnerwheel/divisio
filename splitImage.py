import cv2
import numpy as npù
from util import *

if __name__ == '__main__':    
    
    images = load_images_from_folder("./test")
    for image in images[2:3]:
        image = image[:,:,0]
        contornus(image,treshold=30)
        break