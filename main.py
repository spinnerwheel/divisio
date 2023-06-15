import argparse

import numpy as np
import skimage as sk
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from columGrowing import *
from util import *

if __name__ == "__main__":
    # definizione delle variabili
    folder = "./dataset"
    parser = argparse.ArgumentParser(description="Divisio")
    parser.add_argument("-aph", type=int, help="alpha trimmed kernel",default=7)
    parser.add_argument("-canny", type=float, help="canny sigma", default=2)
    parser.add_argument("--l",dest="low", type=float, required=False, help="canny low threshold", default=None)
    parser.add_argument("--h",dest="high", type=float, required=False, help="canny high threshold", default=None)
    
    args = parser.parse_args()

    print(WELCOME)

    originals, filenames = load_images_from_folder(folder, return_filenames=True)
    masks = []
    dilate_kernel = sk.morphology.disk(1)
    #dilate_kernel_two = sk.morphology.disk(2)
    dilate_kernel_two = sk.morphology.disk(1)
    erode_kernel = np.ones((1,4), np.uint8)
    ax = -1
    for original in originals:
        print(f"Processing image {len(masks)+1}/{len(originals)}...", end="\r")
        image = original.copy()
        image = cv2.resize(image, (image.shape[1]//4, image.shape[0]//4))
        image = ycbcr_filter(image)[0]
        im = gaussian_filter(image,9)
        im = canny_filter(im,1,90,220)
        im = contornus(im, 0, True)
        
        

    final = []
    labels = []
    i=0

    for mask, original, filename in zip(masks, originals, filenames):
        if (original is not None) and (mask is not None) and (filename is not None):
            im = original[:,:,0] & mask
            """
            print(i)
            i=i+1
            cv2.imshow('image',im)
            cv2.waitKey(0)
            """
            
            mu = sk.measure.moments_central(im)
            nu = sk.measure.moments_normalized(mu)
            res = sk.measure.moments_hu(nu)

            #res = np.zeros(0)

            label = filename.split('.')[0].split("-")[2]
            metà = len(label) // 2 
            prima_metà = label[:metà] 
            seconda_metà = label[metà:]  

            area = getArea(im)
            perimeter = getPerimeter(im)
            div = area/perimeter

            res = np.append(res, area)
            res = np.append(res, perimeter)
            res = np.append(res, div)
   
   
            final.append(res)


    plot_features(final[:2], labels)

    print(len(final))
    n_neighbors=3

    X_train, X_test, y_train, y_test = train_test_split(final, labels, test_size = 0.35)

    #classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    classifier = GaussianNB()

    feature_selector = SelectKBest(k=5) # Seleziona le 4 features più significative

    # Creazione del pipeline che combina il selettore delle features e il classificatore Naive Bayes
    classifier = Pipeline([('feature_selector', feature_selector), ('naive_bayes', classifier)])

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    # stampo la percentuale di accuratezza
    print(f"Accuracy with k={n_neighbors}: {accuracy_score(y_test, y_pred)*100}")

    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = classifier.classes_)

    cm_display.plot()
    plt.show()