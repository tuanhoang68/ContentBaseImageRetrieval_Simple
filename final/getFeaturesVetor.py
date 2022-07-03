from numpy import concatenate
import image2histogram as i2h
import getTexture as gt
import color_code_processing as ccp
import numpy as np


def getFeatureVector(image_path):
    intensity_feature = i2h.Color_Histogram_Intensity(image_path)
    colorCode_feature = ccp.getColorCode(image_path)
    texture_feature = gt.getTextureMatrix(image_path)

    # print(intensity_feature)
    # print(colorCode_feature)
    # print(texture_feature)

    feature_vector = np.concatenate(
        (intensity_feature, colorCode_feature, texture_feature))
    return feature_vector


def main():
    image_path = "CBIR/begin/ff.jpg"
    tmp = getFeatureVector(image_path)
    print(tmp)
    # input("Please Enter to Continue...")


if __name__ == '__main__':
    main()
