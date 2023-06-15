import cv2
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

def load_images(path):
    image_list = []
    images_names = os.listdir(path)
    for image_name in images_names:
        read_path = os.path.join(path,image_name)
        image_list.append(cv2.imread(read_path))
    return image_list, images_names

def save_images(path,image_list,image_name):
    delete_images(path)
    for image,name in zip(image_list,image_name):
        write_path = os.path.join(path,name)
        cv2.imwrite(write_path,image)
        
def delete_images(path):
    images_names = os.listdir(path)
    for image_name in images_names:
        os.remove(os.path.join(path,image_name))

def gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def mean_blur(image,kernel=3):
    return cv2.medianBlur(image, kernel)

def gaussian_blur(image,kernel=3):
    return cv2.GaussianBlur(image, (kernel,kernel), 0)

def bilateral_filter(image,kernel=3):
    return cv2.bilateralFilter(image, kernel, 100, 100)

def median_filter(image,kernel=3):
    return cv2.medianBlur(image, kernel)

def canny_edge(image,sigma = 1,threshold1=100,threshold2=200):
    return cv2.Canny(image, threshold1, threshold2,sigma)

def histogram(image,path):
    img = cv2.calcHist([image], [0], None, [256], [0,256])
    plt.hist(img.ravel(),256,[0,256])
    plt.savefig(path)
    plt.close()

def dilate_image(image,kernel):
    return cv2.dilate(image, kernel, iterations=1)

def erode_image(image,kernel):
    return cv2.erode(image, kernel, iterations=1)

def label_connected_components(image,a):
    analysis = cv2.connectedComponentsWithStats(image,8,cv2.CV_32S)
    (totalLabels, label_ids, values, centroid) = analysis
    output = np.zeros(image.shape, dtype="uint8")
    for i in range(1, totalLabels):
        area = values[i, cv2.CC_STAT_AREA] 
        if (area > a):
            componentMask = (label_ids == i).astype("uint8") * 255
            output = cv2.bitwise_or(output, componentMask)
    return output

def flood_filling(image, seed_point):
    im_floodfill = image.copy()
    h, w = image.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, seed_point, 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    filled_image = image | im_floodfill_inv
    return filled_image
