import cv2
import util
import numpy as np

images = util.load_images_from_folder("./benchmark_images")
img = images[0]

image = util.ycbcr_filter(img)[0]
image = util.gaussian_filter(image, 2)
image = util.canny_filter(image, 1,40,80)

contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Crea un'immagine vuota con lo stesso formato dell'immagine di input
img_fill = np.zeros_like(img)

# Disegna i contorni degli oggetti nell'immagine vuota
cv2.drawContours(img_fill, contours, -1, (255, 255, 255), cv2.FILLED)

# Erodi l'immagine binaria
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(img_fill, kernel, iterations=1)