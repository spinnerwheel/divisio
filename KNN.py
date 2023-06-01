import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from util import *


folder = "./saved"

images, filenames = load_images_from_folder(folder, return_filenames=True)
result = []

for image in images:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]
    num_labels, labels = cv2.connectedComponents(img)
    for i in range(1, num_labels):
        labels[labels == i] = 255
    cv2.imshow("image", labels)
    cv2.waitKey(0)