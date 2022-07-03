import os
import glob
import numpy as np
import separate_background as sb
import getFeaturesVetor as gFV


path_image_original = 'C:/Users/ADM/Desktop/HK_8/He_CSDLDPT/btl_nhom3/Code/CBIR/Image/*.jpg'
path_image_separeted = 'C:/Users/ADM/Desktop/HK_8/He_CSDLDPT/btl_nhom3/Code/CBIR/Image/*_Separate.jpg'
path_feature_vectors = 'data_feature_vectors.npy'
path_feature_vector_names = 'data_feature_vector_names.npy'


def Get_List_Feature_Vector(path_image_separeted):
    feature_vectors = []
    feature_vector_names = []

    for (i, image_file) in path_image_separeted:
        vector = gFV.getFeatureVector(image_file)
        feature_vectors.append(vector)
        feature_vector_names.append(image_file)
        # print(i, image_file)

    # print(feature_vector_names)
    np.save(path_feature_vector_names, feature_vector_names)
    return feature_vectors


def Save_Feature_Vectors():
    sb.Create_Separate_Background_Data_Trainning(path_image_original)

    path = enumerate(glob.iglob(path_image_separeted))
    feature_vectors = Get_List_Feature_Vector(path)
    np.save(path_feature_vectors, feature_vectors)
    # print(feature_vectors)


def Open_Feature_Vectors(path_feature_vectors):
    with open(path_feature_vectors, 'rb') as data:
        feature_vectors = np.load(data)
    # print(feature_vectors)

    return feature_vectors


def main():
    # Save_Feature_Vectors()
    # filepath = 'C:/Users/ADM/Desktop/HK_8/He_CSDLDPT/btl_nhom3/Code/CBIR/final/data_feature_vectors.npy'
    # if os.path.exists(filepath) == True:
    #     print("Save file Success!")
    x = Open_Feature_Vectors(path_feature_vectors)
    print(x)
    # y = Open_Feature_Vectors(path_feature_vector_names)
    # print(y)


if __name__ == '__main__':
    main()
