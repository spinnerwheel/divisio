import os
import cv2
import numpy as np
from utils import *
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from featureExtraction import *

images,names = load_images('./results/')
labels = []
features = []

for image,name in zip(images,names):
    image = image[:,:,0]
    labels.append(name.split('.')[0].split('-')[1])
    
    area = getArea(image)
    perimeter = getPerimeter(image)
    major, minor = get_axis(image)
    
    single_feature = np.array([area,perimeter,major, minor])
    
    features.append(single_feature)

feature_train,feature_test,label_train,label_test = train_test_split(features,labels,test_size=0.3,random_state=957)
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(feature_train,label_train)
y_pred = knn.predict(feature_test)
    
print(accuracy_score(label_test,y_pred))

confusion_matrix = metrics.confusion_matrix(label_test, y_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = knn.classes_,)

cm_display.plot()
plt.show()