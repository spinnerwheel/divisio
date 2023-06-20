import os
import cv2
import matplotlib.pyplot as plt


path = './unnamed/'

name_list = []
image_list = []

unamed_list = os.listdir(path)
name_list = ['cacciavite','forbice','pinza','chiaveinglese','unknown','coltello','pelapatate','forchetta','chiave','molletta']
i = 135
print('0:cacciavite, 1:forbice, 2:pinza, 3:chiaveinglese, 4:unknown, 5:coltello, 6:pelapatate, 7:forchetta, 8:chiave, 9:molletta')
for unamed in unamed_list:
    im = cv2.imread(path+unamed)
    im = cv2.resize(im,(im.shape[1]//4,im.shape[0]//4)) 
    value = int(input())
    cv2.imwrite(f'./new_named/{str(i)}-{name_list[value]}.png',im)
    i+=1