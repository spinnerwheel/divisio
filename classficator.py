import numpy as np
from utils import *
from sklearn.neighbors import KNeighborsClassifier
from featureExtraction import *
from binarization import *

a = ['cacciavite','unknown','brugola','unknown','cacciavite','molletta','unknown','brugola','unknown','pelapatate','coltello','brugola','chiaveinglese','unknown','pelapatate','coltello''unknown','brugola','chiaveinglese','unknown']

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
    train_fetures,train_labels = get_train_features('./results/')
    test_fetures,_,test_images = get_test_features('./multi/')
    
    #plot_features(train_fetures,train_labels)
    
    knn = KNeighborsClassifier(n_neighbors=4,weights='distance')
    #train_fetures,test_fetures, train_labels,label_test = train_test_split(train_fetures,train_labels,test_size=0.3,random_state=5884373)

    knn.fit(train_fetures,train_labels)
    y_pred = knn.predict(test_fetures)
    
    print(f'Accuracy: {accuracy_score(a,y_pred)}')
    
'''  
for feature,test_image in zip(test_fetures,test_images):      
    label,prob = get_label_prob(knn,feature)
    plt.imshow(test_image)
    plt.title(f'Label: {label} Prob: {prob}')
    plt.show()  
'''
    #draw_confusion_matrix(label_test,y_pred,knn)
    
