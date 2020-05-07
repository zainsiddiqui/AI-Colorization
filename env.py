# Environment

from PIL import Image
import PIL
from numpy import asarray
import numpy as np
import matplotlib.pyplot as pyplot
import cv2


class Picture:
    def __init__(self, image_file):
        
        image = Image.open(image_file)
        pic = im.load()
        self.x_length = im.size[0]
        self.y_length = im.size[1]


def main(image_file):
    image = Image.open(image_file)
    #print(image.mode)
    data = asarray(image)
    new = np.zeros((len(data),len(data[0]),3), dtype=np.uint8)

    for x in range(int(len(data))):
        for y in range(int(len(data[0]))):
            if (y < int(len(data[0])/2)):
                new[x][y][0] = data[x][y][0] * .21
                new[x][y][1] = data[x][y][1] * .72
                new[x][y][2] = data[x][y][2] * .07
            '''
            else:
                new[x][y][0] = data[x][y][0]
                new[x][y][1] = data[x][y][1]
                new[x][y][2] = data[x][y][2]
            '''
    #print(data)
    img = Image.fromarray(new)
    img.save('new.jpg')

    render = cv2.imread('new.jpg')

    gray = cv2.cvtColor(render, cv2.COLOR_RGB2GRAY)

    cv2.imshow("Gray", gray)
    cv2.waitKey(0)

    return [data, new]



   



image_file = 'beach.jpg'

main(image_file)

