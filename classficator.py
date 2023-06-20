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
            
        major_axis_length, minor_axis_length = get_axis(image)
        rapp = major_axis_length/minor_axis_length
        compactness = get_compactness(image)
        feature = [compactness,rapp]
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
    
    train_fetures,train_labels = get_train_features('./results/')

    find_best_seed(knn,train_fetures,train_labels,True)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #5 neighbors, distance weights, 0.2 test size, 0.8 train size, seed 299396