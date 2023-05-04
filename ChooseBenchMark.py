#read the out.csv file and parse every row dividing the strings using the comma as separator
#then save the result in a list
import csv
import cv2
import os

out = []
read_path = "C:/Users/massi/OneDrive/Desktop/resized512"

with open("out.csv", "r") as f:
    reader = csv.reader(f)
    out = list(reader)
    
for i in range(len(out)):
    print(f"Saving image {i+1}/{len(out)}...", end="\r")
    image_path = read_path + "/resized" + out[i][2]+".jpg"
    image_name = out[i][0]+"_"+out[i][1]+"_"+out[i][2]+".jpg"
    image = cv2.imread(image_path)
    os.chdir("C:/Users/massi/OneDrive/Desktop/divisio/benchmark_images")
    cv2.imwrite(image_name, image)