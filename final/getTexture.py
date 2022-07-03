import numpy as np
import skimage
from skimage import color, img_as_ubyte
import matplotlib.image as mpimg


def RGB_to_Gray(image_path):
    img = mpimg.imread(image_path)
    grayImage = np.zeros(img.shape)
    R = np.array(img[:, :, 0])
    G = np.array(img[:, :, 1])
    B = np.array(img[:, :, 2])

    R = (R * .299)
    G = (G * .587)
    B = (B * .114)

    Avg = (R+G+B)
    grayImage = img.copy()

    for i in range(3):
        grayImage[:, :, i] = Avg

    return grayImage


def GetCoMatrix(img):
    gray = color.rgb2gray(img)
    image = img_as_ubyte(gray)

    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144,
                    160, 176, 192, 208, 224, 240, 255])  # 16-bit
    inds = np.digitize(image, bins)

    max_value = inds.max()+1
    matrix_coocurrence = skimage.feature.graycomatrix(inds, [1], [7*np.pi/4],
                                                      levels=max_value, normed=False, symmetric=False)

    return matrix_coocurrence


# GLCM properties

def contrast_feature(matrix_coocurrence):
    contrast = skimage.feature.graycoprops(matrix_coocurrence, 'contrast')
    return contrast


def energy_feature(matrix_coocurrence):
    energy = skimage.feature.graycoprops(matrix_coocurrence, 'energy')
    return energy


def getTextureMatrix(image_path):
    img_zip = RGB_to_Gray(image_path)
    matrix_coocurrence = GetCoMatrix(img_zip)

    energy = energy_feature(matrix_coocurrence)
    contrast = contrast_feature(matrix_coocurrence)

    feature_vector = np.concatenate((energy[0], contrast[0]))
    return feature_vector


def main():
    image_path = "ff.jpg"
    print(getTextureMatrix(image_path))
    # input("Please Enter to Continue...")


if __name__ == '__main__':
    main()
