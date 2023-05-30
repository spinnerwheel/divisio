import math

import numpy as np
from compute import *
from util import *
import argparse
import skimage as sk 
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from skimage.color import rgb2gray


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
        

final = []
y = []

for filename in filenames:
    
    label = filename.split("_")[1]
    y.append(label)

for image , imageOriginal in zip(result, images):

    imageOriginal = cv2.resize(imageOriginal, (128,128))

    im = imageOriginal[:,:,0] & image
    cv2.imshow('image',im)

    mu = sk.measure.moments_central(im)
    nu = sk.measure.moments_normalized(mu)
    res = sk.measure.moments_hu(nu)
    final.append(res[0:2])


X_train, X_test, y_train, y_test = train_test_split(final, y, test_size = 0.45)

knn = KNeighborsClassifier(n_neighbors = 3) # 3 vicini più vicini con questo dobbiamo giocarci

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

#print(y_pred) # stampo le predizioni del knn

print("Accuracy with k=3", accuracy_score(y_test, y_pred)*100) # stampo la percentuale di accuratezza

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = knn.classes_)

cm_display.plot()
plt.show()
