import tkinter
from tkinter import *
from PIL import ImageTk, Image
from util import *
import skimage.feature
import os  
import numpy as np

param_string = ''
image_dict = {}
parametrized_image = None

def show_image(image,label):
    image = ImageTk.PhotoImage(image)
    label.configure(image=image)
    label.image = image
    label.pack()
    
def update_string(val,changing_param):
    global param_string
    start_index = param_string.find(changing_param)
    if start_index != -1:
        start_index += len(changing_param)+1
        end_index = start_index
        for char in param_string[start_index:]:
            if char == ',':
                break
            end_index += 1
        param_string = param_string[:start_index] + str(val) + param_string[end_index:]
    else:
        param_string += changing_param + '=' + str(val) + ','
        
def get_param(param):
    global param_string
    start_index = param_string.find(param)
    if start_index != -1:
        start_index += len(param)+1
        end_index = start_index
        for char in param_string[start_index:]:
            if char == ',':
                break
            end_index += 1
        return int(param_string[start_index:end_index])
    else:
        return None

def update_alpha(val):
    global parametrized_image
    update_string(val,'alpha')
    if np.any(image_dict.get(param_string) == None):
        image = alpha_trimmed(parametrized_image,int(val))
        image_dict[param_string] = image
    else:
        image = image_dict[param_string]
        
    image = Image.fromarray(image)
    parametrized_image = image
    show_image(image,vlabel)

def update_gauss(val):
    global parametrized_image
    update_string(val,'gauss')
    if np.any(image_dict.get(param_string) == None):
        image = gaussian_filter(parametrized_image,float(val))
        image_dict[param_string] = image
    else:
        image = image_dict[param_string]
        
    image = Image.fromarray(image)
    parametrized_image = image
    show_image(image,vlabel)
    
def reset_image():
    parametrized_image = ycbcr_filter(images[0])[0]
    show_image(parametrized_image,vlabel)
    
root = Tk()
vlabel=Label(root)

images = load_images_from_folder('./benchmark_images/')
parametrized_image = ycbcr_filter(images[0])[0]
show_image(parametrized_image,vlabel)

alpha_scale = Scale(root, from_=1, to=10, resolution=1, orient=HORIZONTAL, label="Alpha",length=300, command=update_alpha)
alpha_scale.pack(anchor=CENTER)
gauss_scale = Scale(root, from_=0, to=2, resolution=0.05, orient=HORIZONTAL, label="Gauss",length=300, command=update_gauss)
gauss_scale.pack(anchor=CENTER)

reset_button = Button(root, text="Reset", command=reset_image)
reset_button.pack(anchor=CENTER)

root.mainloop()