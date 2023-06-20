import cv2
from binarization import *
import numpy as np

def get_area(image):
    area = cv2.countNonZero(image)
    return area

def get_perimeter(image):
    std_image = image.copy()
    dil_im = dilate_image(image,np.ones((3,3),np.uint8))
    perimeter = dil_im - std_image
    return cv2.countNonZero(perimeter)

def getORB(img): 
    orb = cv2.ORB_create()
    _, descriptors = orb.detectAndCompute(img, None) 
    return  np.mean(descriptors)

def get_convex_feature(img):
    convex_im = draw_convex_hull(img)
    convex_area = get_area(convex_im)
    area = get_area(img)
    return convex_area-area

def get_hu_moments(img):
    moments = cv2.moments(img)
    hu_moments = cv2.HuMoments(moments)
    return hu_moments    
    
def draw_convex_hull(img):
    image = img.copy()
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return image
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    hull_img = np.zeros_like(image)
    cv2.drawContours(hull_img, [hull], 0, 255, -1)
    return hull_img

def get_axis(img):
    image = img.copy()
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0,0
    contour = max(contours, key=cv2.contourArea)
    ellipse = cv2.fitEllipse(contour)
    major_axis = max(ellipse[1])
    minor_axis = min(ellipse[1])
    return major_axis,minor_axis

def get_circularity(image):
    area = get_area(image)
    perimeter = get_perimeter(image)
    return (4*np.pi*area)/(perimeter**2)

def get_compactness(image):
    area = get_area(image)
    major_axis,_ = get_axis(image)
    return (major_axis**2)/area

def excentricity(img):
    major_axis,minor_axis = get_axis(img)
    return np.sqrt(1 - (minor_axis**2)/(major_axis**2))