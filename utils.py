import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

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


def multi_label_connected_components(image,a):
    save = False
    output_images = []
    analysis = cv2.connectedComponentsWithStats(image,8,cv2.CV_32S)
    (totalLabels, label_ids, values, centroid) = analysis
    output = np.zeros(image.shape, dtype="uint8")
    for i in range(1, totalLabels):
        area = values[i, cv2.CC_STAT_AREA] 
        if (area > a):
            save = True
            componentMask = (label_ids == i).astype("uint8") * 255
            output = cv2.bitwise_or(output, componentMask)
        if(save):
            output_images.append(output)
            save = False
        output = np.zeros(image.shape, dtype="uint8")
    return output_images

def draw_confusion_matrix(Y_test,y_pred,knn):
    confusion_matrix = metrics.confusion_matrix(Y_test, y_pred)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = knn.classes_,)

    cm_display.plot()
    plt.show()
    
    
def find_best_seed(knn,train_fetures,train_labels,seeds,show=False):
    accuracy_scores = []
    for seed in seeds:
        X_train,X_test, Y_train,Y_test = train_test_split(train_fetures,train_labels,test_size=0.3,random_state=seed)
        knn.fit(X_train,Y_train)
        y_pred = knn.predict(X_test)
        accuracy_scores.append(accuracy_score(Y_test,y_pred))
    if show:       
        print("Accuracy: %0.2f (+/- %0.2f)" % (np.mean(accuracy_scores), np.std(accuracy_scores) * 2))
        print("Max Accuracy: %0.2f" % (np.max(accuracy_scores))+ " with random_state: " + str(seeds[(np.argmax(accuracy_scores))]))
        print("Min Accuracy: %0.2f" % (np.min(accuracy_scores))+ " with random_state: " + str(seeds[(np.argmin(accuracy_scores))]))
        
    return seeds[(np.argmax(accuracy_scores))]

def get_label_prob(knn,feature):
        y_pred = knn.predict_proba(np.reshape(feature,(1,-1)))
        label = knn.classes_[np.argmax(y_pred)]
        best_prob = np.max(y_pred)
        return label,best_prob

    