import cv2
from utils import *
import numpy as np

def getArea(image):
    area = cv2.countNonZero(image)
    return area

def getPerimeter(image):
    std_image = image.copy()
    dil_im = dilate_image(image,np.ones((3,3),np.uint8))
    perimeter = dil_im - std_image
    return cv2.countNonZero(perimeter)
    
    