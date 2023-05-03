from skimage.transform import resize
from skimage.io import imread, imread_collection, imshow_collection, show, imsave

import os

base = "/home/harold/Documents/university/elaborazione-immagini/divisio"
# nome della directory da cui recuperare le immagini
dir = "benchmark_images"
# nome della directory in cui vuoi salvare il file
save_dir = "resized512"
path = os.path.join(base, dir)
save_path = os.path.join(base, save_dir)

files = os.listdir(path)
print("--| Recap |--")
print(f"Path assoluta della directory: {path}")
print(f"Numero di file all'interno: {len(files)}")
print(f"Path assoluta dove verranno salvati i files: {save_path}")

if input("Sicuro di voler continuare? [Y/n]: ") not in ["y", "Y", ""]:
    exit()

paths = [os.path.join(path, f) for f in files]
imgs = imread_collection(paths)
resizeds = [resize(img, [512, 512]) for img in imgs]

for f,r in zip(files, resizeds, strict=True):
    imsave(os.path.join(save_path,f), r)

