from sklearn.cluster import KMeans
from functools import partial

import time
import numpy as np
import matplotlib.pyplot as plt

img_filename = 'ocean.jpg'
img_format = img_filename.split('.')[1]

img = plt.imread(img_filename)
img_height = img.shape[0]
img_width = img.shape[1]
pixel_size = img.shape[2]

pixels = np.reshape(img, (img_height * img_width, pixel_size))

def get_cluster_center(label, centers):
    return centers[label]


def add_image(fig, img, pos, k):
    ax = fig.add_subplot(2, 2, pos)
    if k == 0:
        ax.set_title('original')
    else:
        ax.set_title('k={}'.format(k))
    ax.set_xticks([])
    ax.set_yticks([])

    if (img_format == 'jpg') or (img_format == 'jpeg'):
        plt.imshow(img.astype('uint8'))
    elif img_format == 'png':
        plt.imshow(img)


fig = plt.figure(num="K-Means")
add_image(fig, img, 1, 0)

ks = np.array([3, 5, 7])
for i, k in enumerate(ks):
    start_time = time.time()

    kmeans = KMeans(n_clusters=k, random_state=10).fit(pixels)
    kmeans_img = map(
        partial(get_cluster_center, centers=kmeans.cluster_centers_), kmeans.labels_)
    kmeans_img = np.array(list(kmeans_img))
    kmeans_img = np.reshape(kmeans_img, (img_height, img_width, pixel_size))
    add_image(fig, kmeans_img, i + 2, k)
    
    end_time = time.time()
    print('k = {}, time spent: {}'.format(k, end_time - start_time))

plt.show()