import numpy as np
from utils import *
from sklearn.neighbors import KNeighborsClassifier
from featureExtraction import *
from binarization import *
from matplotlib import pyplot as plt
import cv2

def get_scale_invariant_features(images,names,flag):         #calcola le feature di ogni immagine e le ritorna insieme alle label
    labels = []
    features = []
    
    for image,name in zip(images,names):
        labels.append(name.split('.')[0].split('-')[1])
        if len(image.shape) == 3:
            image = image[:,:,0]
        
        if flag:
            major_axis_length, minor_axis_length = get_axis(image)
            rapp_2 = major_axis_length/minor_axis_length
            compactness = get_compactness(image)
            circularity = get_circularity(image)
            feature = [rapp_2,circularity,compactness]
            features.append(feature)
        else:
            area = get_area(image)
            perimeter = get_perimeter(image)
            ratio = area/perimeter**2
            convex = get_convex_feature(image)
            feature = [area,perimeter,ratio,convex]
            features.append(feature)
            
    return features,labels

def get_features(read_path,flag):
    images,names = load_images(read_path)
    features,label = get_scale_invariant_features(images,names,flag)
    return features,label

if __name__ == '__main__':
    knn = KNeighborsClassifier(n_neighbors=5,weights='distance')
    flag = True
    features,labels = get_features('./results/',flag)
    if flag:
        seed = 606533
    else:
        seed = np.random.randint(0,100000)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=seed)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    print("Accuracy: %0.2f" % (accuracy_score(y_test,y_pred)))
    draw_confusion_matrix(y_test,y_pred,knn)
    