from sklearn.cluster import KMeans
from functools import partial

import time
import numpy as np
import matplotlib.pyplot as plt

img_filename = 'example.png' # replace with the image you like
img_format = img_filename.split('.')[1]

img = plt.imread(img_filename)
img_height = img.shape[0]
img_width = img.shape[1]
pixel_size = img.shape[2] # check rgb or rgba

pixels = np.reshape(img, (img_height * img_width, pixel_size))

def get_cluster_center(label, centers): # get correspond center
    return centers[label]


def add_image(fig, img, pos, k):
    ax = fig.add_subplot(2, 2, pos) # add image to 2*2 figure
    if k == 0:
        ax.set_title('original')
    else:
        ax.set_title(f'k = {k}') # set image title
    ax.set_xticks([]) # remove x ticks
    ax.set_yticks([]) # remove y ticks

    if (img_format == 'jpg') or (img_format == 'jpeg'):
        plt.imshow(img.astype('uint8')) # imshow only supports int 0-255 or float 0-1
    elif img_format == 'png':
        plt.imshow(img)


fig = plt.figure(num="K-Means") # set figure name
add_image(fig, img, 1, 0) # add original image to figure

ks = np.array([3, 5, 7]) # set k value to do kmeans
for i, k in enumerate(ks):
    start_time = time.time()

    kmeans = KMeans(n_clusters=k, random_state=10).fit(pixels) # make kmeans model
    kmeans_img = map(
        partial(get_cluster_center, centers=kmeans.cluster_centers_), kmeans.labels_) # set result array by centers and labels
    kmeans_img = np.array(list(kmeans_img)) # transfer memory address to array
    kmeans_img = np.reshape(kmeans_img, (img_height, img_width, pixel_size))
    add_image(fig, kmeans_img, i + 2, k) # add result image to figure
    
    end_time = time.time()
    print(f'k = {k}, time spent: {end_time - start_time}') # print kmeans spent time

print('Done!')
plt.show()