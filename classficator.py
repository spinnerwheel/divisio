import numpy as np
from utils import *
from sklearn.neighbors import KNeighborsClassifier
from featureExtraction import *
from binarization import *
from matplotlib import pyplot as plt
import cv2

def get_features(images,names):
    labels = []
    features = []
    
    for image,name in zip(images,names):
        labels.append(name.split('.')[0].split('-')[1])
        if len(image.shape) == 3:
            image = image[:,:,0]
            
        area = get_area(image)
        perimeter = get_perimeter(image)
        rapp = area/perimeter**2
        convex = get_convex_feature(image)
        major_axis_length, minor_axis_length = get_axis(image)
        rapp_2 = major_axis_length/minor_axis_length
        compactness = get_compactness(image)
        feature = [compactness,rapp_2]
        features.append(feature)    
        
    return features,labels

def get_train_features(read_path):
    images,names = load_images(read_path)
    features,label = get_features(images,names)
    return features,label

def get_test_features(read_path):
    images,names = load_images(read_path)
    images,names = multi_images_binarization(images,names)
    features,labels = get_features(images,names)
    return features,labels,images


if __name__ == '__main__':
    knn = KNeighborsClassifier(n_neighbors=5,weights='distance')
    
    fetures,labels = get_train_features('./results/')
    
    find_best_seed(knn,fetures,labels,True)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #5 neighbors, distance weights, 0.2 test size, 0.8 train size, seed 299396
    
    #5 neighbors, distance weights, 0.2 test size, 0.8 train size, seed 524748 area e tutte cose
    
    #5 neighbors, distance weights, 0.2 test size, 0.8 train size, seed 863767 area 95%
    
    #5 neighbors, distance weights, 0.2 test size, 0.8 train size, seed 819055 area e invarianti 95%
    
    #5 neighbors, distance weights, 0.2 test size, 0.8 train size, seed 819055 invarianti 88% e 2 features
    
    