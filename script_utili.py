import cv2
import numpy as np

def gaussian_filter(img, kernel_size=7, sigma=1.2):
    # Apply a Gaussian Blur
    blur_gray = cv2.GaussianBlur(img,(kernel_size, kernel_size), sigma)
    return blur_gray

def hsv_filter(img):
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return [hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]]

def ycbcr_filter(img):
    # Convert to YCrCb color space and separate the Y channel
    ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    return [ycbcr[:,:,0], ycbcr[:,:,1], ycbcr[:,:,2]]

def binarize(img):
    # Threshold the image
    ret, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return binary

def histogram_equalization(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    equalized_rgb = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
    return equalized_rgb

#apply a dilate filter to the image to make the white lines more visible using a disk of radius 3
def dilate_filter(img,kernel):
    dilation = cv2.dilate(img,kernel,iterations = 1)
    return dilation

def erode_filter(img,kernel):
    closing = cv2.erode(img, kernel,iterations=1)
    return closing

def shadow_removal(image):
    # converte l'immagine in scala di grigi
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # applica un filtro di mediana per ridurre il rumore
    gray = cv2.medianBlur(gray, 5)

    # applica un filtro di Sobel per individuare i bordi dell'immagine
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)

    # normalizza l'immagine
    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    # applica un algoritmo di segmentazione per separare l'ombra dall'oggetto
    _, thresh = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    mask = cv2.erode(mask, None, iterations=4)
    mask = cv2.dilate(mask, None, iterations=4)

    # rimuove l'ombra dall'immagine
    return cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)

def alpha_trimmed(image,kernel=7):
    tmp = np.copy(image)
    for x in range(0,len(tmp)-kernel,kernel):
        for y in range(0,len(tmp[0])-kernel,kernel):
            textel = tmp[x:x+kernel,y:y+kernel]
            textel = np.sort(textel,axis=None)
            textel = textel[1:-1]
            tmp[x:x+kernel,y:y+kernel] = np.mean(textel)
    return tmp