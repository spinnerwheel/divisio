import tkinter
from tkinter import *
from PIL import ImageTk, Image
from util import *
from ImageParam import *

param_string = ''
image_dict = {}
parametrized_image = None

def show_image(image,label):
    image = ImageTk.PhotoImage(image)
    label.configure(image=image)
    label.image = image
    label.pack()

def update_alpha(val):
    ip.set_param(val,'alpha')
    show_image(ip.current_image,vlabel)
    
def update_gauss(val):
    ip.set_param(val,'gauss')    
    show_image(ip.current_image,vlabel)

def update_lb_canny(val):
    ip.set_param(val,'lb_canny')
    show_image(ip.current_image,vlabel)
        
def reset_image():
    ip.reset_image()
    show_image(ip.current_image,vlabel)
    
root = Tk()
vlabel=Label(root)

images = load_images_from_folder('./benchmark_images/')
parametrized_image = ycbcr_filter(images[0])[0]
show_image(parametrized_image,vlabel)

ip = ImageParam(parametrized_image)

alpha_scale = Scale(root, from_=2, to=11, resolution=1, orient=HORIZONTAL, label="Alpha",length=300, command=update_alpha)
alpha_scale.pack(anchor=CENTER)

gaussian_scale = Scale(root,from_=0,to=4,resolution=0.02,orient= HORIZONTAL,label = "Gaussian", length=300,command=update_gauss)
gaussian_scale.pack(anchor=CENTER)

lb_canny_scalar = Scale(root,from_=0,to=100,resolution=1,orient= HORIZONTAL,label = "Lower BoundCanny Scalar", length=300,command=update_lb_canny)
lb_canny_scalar.pack(anchor=CENTER)

reset_button = Button(root, text="Reset", command=reset_image)
reset_button.pack(anchor=CENTER)

root.mainloop()