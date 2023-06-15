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

def getORB(img): 
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(img, None) 
    return  np.mean(descriptors)

def convex_feature(img):
    convex_im = draw_convex_hull(img)
    convex_area = getArea(convex_im)
    area = getArea(img)
    return convex_area/area
    
    
def draw_convex_hull(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return img
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    hull_img = np.zeros_like(img)
    cv2.drawContours(hull_img, [hull], 0, 255, -1)
    return hull_img

def get_axis(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w)/h
        if aspect_ratio > 1:
            major_axis = aspect_ratio * np.sqrt(area / np.pi)
            minor_axis = np.sqrt(area / np.pi)
        else:
            major_axis = np.sqrt(area / np.pi)
            minor_axis = aspect_ratio * np.sqrt(area / np.pi)
    return major_axis,minor_axis