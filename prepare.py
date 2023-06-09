from skimage.transform import resize
from skimage.io import imread, imread_collection, imshow_collection, show, imsave

import os
import sys
import csv

from util import load_images_from_folder

i = 0
BASE = ""
# nome della directory da cui recuperare le immagini
DIR = "dataset"
# nome della directory in cui vuoi salvare il file
SAVE_DIR = "temp"
path = os.path.join(BASE, DIR)
save_path = os.path.join(BASE, SAVE_DIR)

if not os.path.isdir(save_path):
    os.mkdir(save_path)

names = []
with open("out.csv", "r") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        names.append(row)

images, filenames = load_images_from_folder(path, return_filenames=True)

print("--| Recap |--")
print(f"Path assoluta della directory: {path}")
print(f"Numero di file all'interno: {len(filenames)}")
print(f"Path assoluta dove verranno salvati i files: {save_path}")

if input("Sicuro di voler continuare? [Y/n]: ") not in ["y", "Y", ""]:
    sys.exit()

for img, f in zip(images, filenames):
    i+=1
    print(f"Resizing, renaiming and saving image {i}/{len(images)}...", end="\r")
    for row in names:
        if f"resized{row[0]}.jpg" == f:
            filename = f"{row[0]}-{row[1]}-{row[2]}.jpg"
            names.remove(row)
            break
    resized = resize(img, [128, 128])
    resized = resized * 255
    out = resized.astype("uint8")
    imsave(f"{save_path}/{filename}", out, check_contrast=False)
if len(names) != 0:
    print("Something sus is going on...")
