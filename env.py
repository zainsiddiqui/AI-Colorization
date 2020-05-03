# Environment

from PIL import Image
import numpy as np

class Picture:
    def __init__(self, image_file):
        
        im = Image.open(image_file)
        pic = im.load()
        self.x_length = im.size[0]
        self.y_length = im.size[1]

Picture("beach.jpg")

