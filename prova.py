from util import *
import numpy as np
from PIL import Image


#create a new image of shape 7x7 and fill it with zeros besides the center
img = np.zeros((7,7), dtype=np.uint8)

img[3,3] = 255
img[3,2] = 255
img[3,4] = 255

img[2,3] = 255
img[4,3] = 255

a = getArea(img)
p = getPerimeter(img)

print(a)
print(p)
