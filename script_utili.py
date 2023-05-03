import cv2

def gaussian_filter(img, kernel_size=7, sigma=1.2):
    # Apply a Gaussian Blur
    blur_gray = cv2.GaussianBlur(img,(kernel_size, kernel_size), sigma)
    return blur_gray

def hsv_filter(img):
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv

def ycbcr_filter(img):
    # Convert to YCrCb color space and separate the Y channel
    ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    return ycbcr

def binarize(img):
    # Threshold the image
    ret, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return binary

#apply a dilate filter to the image to make the white lines more visible using a disk of radius 3
def dilate_filter(img,kernel):
    dilation = cv2.dilate(img,kernel,iterations = 1)
    return dilation

def histogram_equalization(img):
    # Apply Histogram Equalization
    equ = cv2.equalizeHist(img)
    return equ

def close_filter(img,kernel):
    closing = cv2.erode(img, kernel,iterations=1)
    return closing