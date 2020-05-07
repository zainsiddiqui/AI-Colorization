# Improved Agent
from PIL import Image
import PIL
from numpy import asarray
import numpy as np


class ConvNeuralNetwork:

    def __init__(self, image_file, data, new):
        self.input = image_file 
        self.data = data
        self.new = new
        
    def get_size(image):
        return image.size

    def rgbgrey(rgb):
        return np.dot(rgb[...,:3], [.21,.72,.07])

    def generateFilter1(self, numFilters, x, y):
        filterr = np.zeros((numFilters,x,y), dtype=int)
        #print(filterr)
        for k in range(numFilters):
            count = k
            for i in range(x):
                for j in range(y):
                    if (count == 0):
                        filterr[k][i][j] = 1
                        filterr[k][i][j] = 0
                        filterr[k][i][j] = 1
                        count = 1
                    elif (count == 1):
                        filterr[k][i][j] = 0
                        filterr[k][i][j] = 1
                        filterr[k][i][j] = 0
                        count = 2
                    else:
                        filterr[k][i][j] = -1
                        filterr[k][i][j] = -1
                        filterr[k][i][j] = -1
                        count = 0        
        return filterr 

    def generateFilter2(self, numFilters, x, y):
        filterr = np.zeros((numFilters,x,y), dtype=int)
        #print(filterr)
        for k in range(numFilters):
            count = k
            for i in range(x):
                for j in range(y):
                    if (count == 0):
                        filterr[k][i][j] = 2
                        filterr[k][i][j] = 0
                        filterr[k][i][j] = 2
                        count = 1
                    elif (count == 1):
                        filterr[k][i][j] = 0
                        filterr[k][i][j] = -1
                        filterr[k][i][j] = 0
                        count = 2
                    else:
                        filterr[k][i][j] = 1
                        filterr[k][i][j] = 0
                        filterr[k][i][j] = 1
                        count = 0        
        return filterr 

        def generateFilter3(self, numFilters, x, y):
        filterr = np.zeros((numFilters,x,y), dtype=int)
        #print(filterr)
        for k in range(numFilters):
            count = k
            for i in range(x):
                for j in range(y):
                    if (count == 0):
                        filterr[k][i][j] = 0
                        filterr[k][i][j] = -1
                        filterr[k][i][j] = 0
                        count = 1
                    elif (count == 1):
                        filterr[k][i][j] = 2
                        filterr[k][i][j] = 0
                        filterr[k][i][j] = 2
                        count = 2
                    else:
                        filterr[k][i][j] = 0
                        filterr[k][i][j] = 1
                        filterr[k][i][j] = 0
                        count = 0        
        return filterr 

    def applyFilter(self, filterr, data):
        #print((filterr[0]))
        new = np.zeros((len(filterr[0]),len(filterr[0][0])), dtype = int )
        print(new[0][0])
        #print(new[1][1])
        for i in range(len(data) - len(filterr[0])):
            for j in range(len(data[0]) - len(filterr[0][0])):
                summ = 0
                for x in range(len(filterr[0])):
                    for y in range(len(filterr[0][0])):
                        summ = data[i + x][j + y] * filterr[x][y]
            new[i][j] = summ
        print(new)
        return new



    def pooling(self, data, filterr, new):
        result = np.zeros(len(data) / 4, len(data[0] / 4))
        countx = 0
        county = 0
        maxx = 0
        for x in range(len(data) - 2):
            for y in range(len(data) - 2):
                maxx = 0
                for i in range(x + i):
                    for j in range(y + j):
                        result[countx][county] = result[countx][county] + data[i][j]
                        current = data[i][j]
                        if (current > maxx):
                            max = current
                        result[countx][county] = max
        return [result, max_pooling(result)] 
    
    def define_max_pooling(new):
        maxX = 0
        maxY = 0
        summ = 0
        maxx = 0
        for x in range(len(new) - 2):
            for y in range(len(new[0]) - 2):
                summ = new[x][y] + new[x+1][y] + new[x][y+1] + new[x+1][y+1]
                if (summ > maxx)
                    maxx = summ
                    maxX = x
                    maxY = y
            summ = 0
        return [maxx, maxX, maxY]

    def ReLU_activation(arr, act):
        if (act == 1):
            arr = np.maximum(0, arr)
        else:
            for x in range(len(arr)):
                for y in range(len(arr)):
                    if (arr[x][y] < 0 and act = 0):
                        arr[x][y] = arr[x][y] * -1

        return arr


    def soft_max(data, filterr, new):
        for x in range(len(filterr)):
           current = filterr[x]
            for i in current:
                result = np.zeros(1, data[x] / 100)
        return result
    
    def activation_data(arr):
        result = 1 / (1 + np.exp(-arr))
        return result
    
    def applyFilterExtra(self, filterr, data):
        #print((filterr[0]))
        new = np.zeros((len(filterr[0]),len(filterr[0][0])), dtype = int )
        for i in range(len(data) - len(filterr[0])):
            for j in range(len(data[0]) - len(filterr[0][0])):
                summ = 0
                for x in range(len(filterr[0])):
                    for y in range(len(filterr[0][0])):
                        summ = data[i + x][j + y] * filterr[x][y]
            new[i][j] = summ
        return new
    
    def organize_result(self, result, original):
        for x in range(len(result)/2):
            for y in range(len(result[0])/2):
                result[x][y] = original[x][y]
        return result




    

# generic
def getData(image_file):
    image = Image.open(image_file)
    data = asarray(image)
    return data

def getNew(data):
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
# Get numpy array of rgb/pixels
data = getData(image_file)
# Get numpy array of specified greyscale values
new = getNew(data)


network = ConvNeuralNetwork(image_file, data, new)
# number of filters, row, column

new = activation_data(new)
filterr = network.generateFilter(3, 3, 3)
new = network.applyFilter(filterr, data)


new = network.applyFilter1(fitlerr, data)

new = network.ReLU_activation(new, 1)

result = pooling(data, filterr, new)[0]

result = soft_max(data, filterr, result)

print(result)





