import numpy as np
from utils import *
from sklearn.neighbors import KNeighborsClassifier
from featureExtraction import *
from binarization import *

def get_features(images,names):
    labels = []
    features = []
    
    
    for image,name in zip(images,names):
        single_feature = np.zeros(0)
        labels.append(name.split('.')[0].split('-')[1])
        if len(image.shape) == 3:
            image = image[:,:,0]
            
        area = getArea(image)             
        perimeter = getPerimeter(image)
        rapporto = area/perimeter**2
        compactness = (4 * np.pi * area) / (perimeter ** 2)
        convex_feature = get_convex_feature(image)

        hu_moments = get_hu(image)
        #single_feature = np.array([rapporto, convex_feature])
        for val in hu_moments:
            single_feature = np.append(single_feature, val)

        features.append(single_feature)
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
    #test_fetures,test_labels,test_images = get_test_features('./multi/')
    
    knn = KNeighborsClassifier(n_neighbors=4, weights='distance')
    #best_seed = find_best_seed(knn,train_fetures,train_labels, random_seeds, show = True)
    train_fetures,test_fetures, train_labels,label_test = train_test_split(train_fetures,train_labels,test_size=0.4,random_state=0)

    knn.fit(train_fetures, train_labels)
    """
    for feature,name,test_image in zip(test_fetures,test_labels,test_images):      
        label,prob = get_label_prob(knn,feature)
        plt.imshow(test_image)
        plt.title(f'Label: {label} Prob: {prob}')
        plt.show() 
        """

    y_pred = knn.predict(test_fetures)

    print("Accuracy with k=3", accuracy_score(label_test, y_pred)*100) # stampo la percentuale di accuratezza

    confusion_matrix = metrics.confusion_matrix(label_test, y_pred)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = knn.classes_)

    cm_display.plot()
    plt.show()
    
