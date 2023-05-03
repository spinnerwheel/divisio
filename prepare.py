from skimage.transform import resize
from skimage.io import imread, imread_collection, imshow_collection, show, imsave

import os

i = 0
base = "C:/Users/massi/OneDrive/Desktop"
# nome della directory da cui recuperare le immagini
dir = "new_dataset"
# nome della directory in cui vuoi salvare il file
save_dir = "resized512"
path = os.path.join(base, dir)
save_path = os.path.join(base, save_dir)

if not os.path.isdir(save_path):
    os.mkdir(save_path)

files = os.listdir(path)
print("--| Recap |--")
print(f"Path assoluta della directory: {path}")
print(f"Numero di file all'interno: {len(files)}")
print(f"Path assoluta dove verranno salvati i files: {save_path}")

if input("Sicuro di voler continuare? [Y/n]: ") not in ["y", "Y", ""]:
    exit()

paths = [os.path.join(path, f) for f in files]
imgs = imread_collection(paths)

for img, f in zip(imgs, files):
    i+=1
    print(f"Resizing and saving image {i}/{len(imgs)}...", end="\r")
    resized = resize(img, [512, 512])
    resized = resized * 255
    sus = resized.astype("uint8")
    imsave(f"{save_path}/resized{i}.jpg", sus, check_contrast=False)
