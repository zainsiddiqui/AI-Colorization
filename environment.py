import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt


def crop_bottom_half(image):
    cropped_img = image[int(image.shape[0])/2:int(image.shape[0])]
    return cropped_img

original_image = cv2.imread("beach.jpg")
img=cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
vectorized = img.reshape((-1,3))
vectorized = np.float32(vectorized)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 5
attempts=10
ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
# Array of 5 representative RGB Values
repColors = center
print(repColors)
result_image = res.reshape((img.shape))





figure_size = 15
plt.figure(figsize=(figure_size/2,figure_size/2))
plt.subplot(1,2,1),plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(result_image)
plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])
plt.show()