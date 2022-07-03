import numpy as np
import glob
import os
import cv2 as cv
from matplotlib import pyplot as plt

# Đuôi file đã tách nền, thêm vào path để phân biệt với file gốc
end_name = "_Separate.jpg"

# Bước tách vật thể khỏi background (Tìm hiểu thêm GrabCut Arlogithm)
# Tham khảo thêm tài liệu thuật toán tại:
# https://docs.opencv.org/3.4/d8/d83/tutorial_py_grabcut.html
# hoặc
# https://www.sicara.fr/blog-technique/grabcut-for-automatic-image-segmentation-opencv-tutorial


def Separate_Background(img_path):
    img = cv.imread(img_path)  # Ảnh xử lý
    img_display = cv.imread(img_path)  # Ảnh hiển thị khi truy vấn
    height, width, _ = img.shape
    cv.setRNGSeed(0)
    mask = np.zeros((height, width), np.uint8)  # Mặt nạ thứ nhất

    bgdModel = np.zeros((1, 65), np.float64)  # 2 lớp mặc định của thuật toán
    fgdModel = np.zeros((1, 65), np.float64)

    # Hình chữ nhật xác định khung hình chọn làm foreground
    # rect = (x, y, w, h)
    # Trong đó (x, y) là tọa độ góc trái trên cùng của khung, (w, h) là chiều ngang và độ cao của ảnh
    rect = (20, 50, 450, 500)

    # Chạy thuật toán
    # Thuật toán có dạng:
    #   cv.grabCut (img, mask, rect, bgdModel, fgdModel, iterCount, mode)
    #       - img - Hình ảnh đầu vào
    #       - mask - Đây là hình ảnh mặt nạ trong đó chúng tôi chỉ định khu vực nào
    #               là hậu cảnh, tiền cảnh hoặc có thể là hậu cảnh / nền trước, v.v. Nó được
    #               thực hiện bởi các cờ sau, cv.GC_BGD , cv.GC_FGD , cv.GC_PR_BGD ,
    #               cv.GC_PR_FGD hoặc đơn giản là vượt qua 0,1,2,3 sang hình ảnh.
    #       - rect - Là tọa độ của hình chữ nhật bao gồm đối tượng nền trước ở định
    #               dạng (x, y, w, h).
    #       - bdgModel , fgdModel - Đây là các mảng được thuật toán sử dụng trong nội
    #               bộ. Bạn chỉ cần tạo hai mảng 0 kiểu np.float64 có kích thước (1,65).
    #       - iterCount - Số lần lặp lại mà thuật toán sẽ chạy.
    #       - mode - Nó phải là cv.GC_INIT_WITH_RECT hoặc cv.GC_INIT_WITH_MASK hoặc kết
    #               hợp để quyết định xem chúng ta đang vẽ hình chữ nhật hay các nét chạm
    #               cuối cùng (chắc là viển ngoài foreground).
    cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

    # Sửa đổi hình ảnh mặt nạ. Trong hình ảnh mặt nạ mới, các pixel sẽ được đánh dấu bằng
    # bốn cờ biểu thị background/foreground như đã chỉ định ở trên. Vì vậy, chúng tôi sửa
    # đổi mặt nạ sao cho tất cả 0 pixel và 2 pixel được đưa về 0 (tức là background) và
    # tất cả 1 pixel và 3 pixel được đặt thành 1 (tức là pixel foreground).
    mask2 = np.where((mask == cv.GC_PR_BGD) | (
        mask == cv.GC_BGD), 0, 1).astype('uint8')

    # Bây giờ mặt nạ cuối cùng của chúng tôi đã sẵn sàng. Chỉ cần nhân nó với hình ảnh đầu vào
    # để có hình ảnh được phân đoạn.
    img = img*mask2[:, :, np.newaxis]

    # Hiển thị ảnh
    plt.imshow(img_display)
    full_path = "1.jpg"  # Đường dẫn (lưu trong folder final luôn)
    plt.savefig(full_path)  # Lưu ảnh
    os.startfile(full_path)  # Mở ảnh

    # Lưu ảnh đã xóa nền
    plt.imshow(img)
    # Điều kiện dưới đây để tránh chạy nhiều lần bị tạo ra nhiều file xóa nền khi chạy lại
    # kiểu như *_Separate_Separate_Separate.jpg
    if(img_path.find(end_name) >= 0):
        file_name = img_path
    else:
        file_name = img_path.replace(".jpg", end_name)
    plt.savefig(file_name, transparent=True)


# Thêm file xóa nền cho tất cả các ảnh trong database


def Create_Separate_Background_Data_Trainning(path):
    source_images = enumerate(glob.iglob(path))
    # source_images = enumerate(glob.iglob(path+end_name)) # Cái này là do lỗi xóa nền 2 lần sẽ thu về ảnh gốc bị xóa nền (tốt hơn)

    for (i, image_file) in source_images:
        Separate_Background(image_file)


# Demo module


def main():
    path = "CBIR/begin/ff.jpg"
    path2 = path.replace(".jpg", end_name)
    Separate_Background(path)
    Separate_Background(path2)
    # input("Please Enter to Continue...")


if __name__ == '__main__':
    main()
