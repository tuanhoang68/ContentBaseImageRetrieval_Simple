# adding Folder_2 to the system path
import image2histogram as i2h
import color_code_processing as ccp
import getFeaturesVetor as FV
from scipy.spatial.distance import cityblock
import numpy as np


def get_distance_manhattan(vector1, vector2):
    return cityblock(vector1, vector2)


def get_vector_distances(list_vector, vector2):
    distances = []
    for vector in list_vector:
        dist = get_distance_manhattan(vector, vector2)
        distances.append(dist)
    return distances


# img1 = "ff.jpg"
# img2 = "ff2.jpg"

# bins1 = FV.getFeatureVector(img1)
# bins2 = FV.getFeatureVector(img2)
# m = 512*512
# # print(cityblock(bins1/m, bins2/m))


# print(bins1)
# print(bins2)
# print(cityblock(bins1, bins2))
