import numpy as np
import matplotlib.image as mpimg


def RGB2ColorCode(array):
    bit_r = f'{array[0]:08b}'
    bit_g = f'{array[1]:08b}'
    bit_b = f'{array[2]:08b}'

    zipcode = bit_r[0] + bit_r[1] + \
        bit_g[0] + bit_g[1] + bit_b[0] + bit_b[1]
    return zipcode


def Binary2Integer(string_binary):
    number_integer = int(string_binary, 2)
    return number_integer


def getColorCode(image_path):
    img = mpimg.imread(image_path)
    Image_Height = img.shape[0]
    Image_Width = img.shape[1]

    bins = np.zeros(64, dtype=int)

    for x in range(0, Image_Height):
        for y in range(0, Image_Width):
            ar = [img[x, y, 0], img[x, y, 1], img[x, y, 2]]
            bit_color_6 = RGB2ColorCode(ar)
            index = Binary2Integer(bit_color_6)
            bins[index] += 1

    sum = np.sum(bins)
    if sum == 0:
        sum = 1
    bins = np.true_divide(bins, sum)  # <=> bins /= sum
    return bins


def main():
    image_path = "ff.jpg"
    print(getColorCode(image_path))
    # input("Please Enter to Continue...")


if __name__ == '__main__':
    main()
