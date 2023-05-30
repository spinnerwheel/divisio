import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics



X, y = make_blobs(n_samples = 100, n_features = 3, centers = 2,cluster_std = 1.5, random_state = 4)
#print(X)# featurs estratte nella forma di array di array(3 feature e 10 oggetti = 1 array che contiene 10 array di 3 elementi)
print(y)# lui sa tutto e ti da la classe di ogni oggetto


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.45)

knn = KNeighborsClassifier(n_neighbors = 3) # 3 vicini più vicini con questo dobbiamo giocarci

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

#print(y_pred) # stampo le predizioni del knn

print("Accuracy with k=2", accuracy_score(y_test, y_pred)*100) # stampo la percentuale di accuratezza

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = knn.classes_)

cm_display.plot()
plt.show()