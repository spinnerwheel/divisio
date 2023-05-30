import skimage as sk 
from util import *
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from skimage.color import rgb2gray

folder = "./saved/"

images, filenames = load_images_from_folder(folder, return_filenames=True)

folder = "./benchmark_images/"

imagesOriginal = load_images_from_folder(folder, return_filenames=False)

result = []

y = []

for filename in filenames:
    
    label = filename.split("_")[1]
    y.append(label)

for image , imageOriginal in zip(images, imagesOriginal):
    image = image[:,:,0]
    cv2.imshow("image", image)

    cv2.waitKey(0)
    imageOriginal = rgb2gray(imageOriginal)

    im = imageOriginal & im
    mu = sk.measure.moments_central(im)
    nu = sk.measure.moments_normalized(mu)
    featur = sk.measure.moments_hu(nu)
    result.append(featur[0:2])

print(result)

X_train, X_test, y_train, y_test = train_test_split(result, y, test_size = 0.45)

knn = KNeighborsClassifier(n_neighbors = 3) # 3 vicini più vicini con questo dobbiamo giocarci

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

#print(y_pred) # stampo le predizioni del knn

print("Accuracy with k=3", accuracy_score(y_test, y_pred)*100) # stampo la percentuale di accuratezza

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = knn.classes_)

cm_display.plot()
plt.show()