import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
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


def Histogram_Computation(Image):

    Image_Height = Image.shape[0]
    Image_Width = Image.shape[1]

    Histogram = np.zeros([256], np.int32)

    for x in range(0, Image_Height):
        for y in range(0, Image_Width):
            Histogram[Image[x, y]] += 1

    return Histogram


def Color_Histogram_Intensity(image_path):
    Input_Image = RGB_to_Gray(image_path)

    bins = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                     15, 16, 17, 18, 19, 20, 21, 22, 23, 24])

    Histogram_GrayScale = Histogram_Computation(Input_Image)

    # print(len(Histogram_GrayScale))

    # Now to print our output Histogram
    x = 0
    for bin in bins:
        bins[bin] = 0
        if x < 240:
            k = x + 10
        else:
            k = x + 16
        while x < k:
            if x == 256:
                break
            bins[bin] += Histogram_GrayScale[x]
            x += 1

    sum = np.sum(bins, dtype=np.float32)
    if sum == 0:
        sum = 1
    bins = np.true_divide(bins, sum)  # <=> bins /= sum
    return bins


def main():
    Color_Histogram_Intensity("ff.jpg")
    # input("Please Enter to Continue...")


if __name__ == '__main__':
    main()
