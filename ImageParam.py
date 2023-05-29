from typing import Any
import numpy as np
from PIL import Image
from util import *

class ImageParam:
    def __init__(self, image):
        self.original_image = image
        self.current_image = image
        self.param_list = []
        self.image_dict = {}

    
if __name__ == '__main__':
    ip = ImageParam(load_images_from_folder('benchmark_images/')[0])
    ip.set_param(1,'alpha')