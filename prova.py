from featureExtraction import *
import cv2
import numpy as np

def get_major_minor_axis_lengths(image):
    # Trova i contorni dell'oggetto nell'immagine
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Calcola la convex hull dell'oggetto
    hull = cv2.convexHull(contours[0])

    # Calcola la lunghezza dell'asse maggiore e minore della convex hull
    major_axis_length, minor_axis_length = 0, 0
    for i in range(len(hull)):
        for j in range(i, len(hull)):
            distance = np.linalg.norm(hull[i] - hull[j])
            if distance > major_axis_length:
                major_axis_length = distance
            elif distance > minor_axis_length:
                minor_axis_length = distance

    return major_axis_length, minor_axis_length

test_img = np.zeros((100,100),np.uint8)

test_img[10:20,10:20] = 255

cv2.imshow("test",test_img)
cv2.waitKey()
print(get_major_minor_axis_lengths(test_img))

