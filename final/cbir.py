import numpy as np
import getFeaturesVetor as gFV
import manhattanDistance as mD
import matplotlib.pyplot as plt
import math
from PIL import Image
import separate_background as sb
import random
import os


end_name = "_Separate.jpg"
length_feature_vector = 91  # Độ dài vector đặc trưng

path_feature_vectors = 'data_feature_vectors.npy'
path_feature_vector_names = 'data_feature_vector_names.npy'
path_image_query = 'CBIR/begin/421705.jpg'
path_image_query_Separated = path_image_query.replace(".jpg", end_name)


def Open_Feature_Vectors(path_feature_vectors):
    with open(path_feature_vectors, 'rb') as data:
        feature_vectors = np.load(data)
    # print(feature_vectors)

    return feature_vectors


def get_display_image_path(feature_vector_path, dis):
    imagesDisplayed = 10  # Số ảnh hiển thị
    # Sắp xếp khoảng cách, lấy ra các index
    ids = np.argsort(dis)[:imagesDisplayed]
    # print(ids)

    nearest_images = [(feature_vector_path[id], dis[id]) for id in ids]

    # for img in nearest_images:
    #     print(img)

    axes = []
    grid_size = int(math.sqrt(imagesDisplayed))
    fig = plt.figure(figsize=(10, 5))

    for id in range(imagesDisplayed):
        draw_image = nearest_images[id]
        axes.append(fig.add_subplot(grid_size, grid_size + 1, id+1))

        axes[-1].set_title(draw_image[1])
        plt.imshow(Image.open(draw_image[0]))

    fig.tight_layout()
    name_image_cbir = random.randint(0, 90000)
    full_path = str(name_image_cbir) + ".jpg"
    plt.savefig(full_path)

    # This function returns a dictionary
    image_query_package = dict()
    image_query_package['full_path'] = full_path
    image_query_package['ids'] = ids
    image_query_package['display_path'] = name_image_cbir

    return image_query_package  # Trả về 1 object


def Processing_Image_Query():
    sb.Create_Separate_Background_Data_Trainning(path_image_query)
    feature_vector_img = gFV.getFeatureVector(path_image_query_Separated)

    return feature_vector_img


def main():
    feature_vectors = Open_Feature_Vectors(path_feature_vectors)
    feature_vector_names = Open_Feature_Vectors(path_feature_vector_names)

    # Processing Image Query
    feature_vector_img = Processing_Image_Query()

    # Distance
    dis = mD.get_vector_distances(feature_vectors, feature_vector_img)

    feature_vector_path = []

    for path in feature_vector_names:
        path = path.replace("_Separate.jpg", ".jpg")

        feature_vector_path.append(path)

    image_query_package = get_display_image_path(feature_vector_path, dis)

    # Display
    os.startfile(image_query_package["full_path"])

    # RELEVANCE FEEDBACK
    print('Choose some photo related to the query photo:')
    photo = input()

    os.remove(str(image_query_package["display_path"])+".jpg")

    answer = []  # Mảng các lựa chọn
    # Tách input thành các số
    for t in photo.split():
        try:
            answer.append(int(t))
        except ValueError:
            pass
    # print(len(answer))

    relevant = np.zeros(length_feature_vector)
    non_relevant = np.zeros(length_feature_vector)
    marker_array = image_query_package["ids"]  # Khởi tạo mảng đánh dấu

    for number in answer:
        image_Related = image_query_package["ids"][number]
        marker_array[number] = -1
        for i in range(0, length_feature_vector):
            relevant[i] += feature_vectors[image_Related][i]
    # print(relevant)

    for index, value in enumerate(marker_array):
        if value > 0:
            # print(feature_vectors[value])
            # print(index, value)
            for i in range(0, length_feature_vector):
                non_relevant[i] += feature_vectors[value][i]

    # print(non_relevant)
    alpha = 1
    beta = 0.5
    gamma = 0.5

    vector_RF = alpha*feature_vector_img + \
        beta * relevant/(len(answer)) - gamma*non_relevant
    print(vector_RF)

    for number in range(0, length_feature_vector):
        if vector_RF[number] < 0:
            vector_RF[number] = 0

    print(vector_RF)

    #############
    dis = mD.get_vector_distances(feature_vectors, vector_RF)
    image_query_package = get_display_image_path(feature_vector_path, dis)

    # Display
    os.startfile(image_query_package["full_path"])
    print('Press any key to Exit!')
    photo = input()
    ##############

    # Delete image in folder
    os.remove(image_query_package["full_path"])
    os.remove("1.jpg")


if __name__ == '__main__':
    main()
