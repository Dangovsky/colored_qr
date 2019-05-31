import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from pyzbar.pyzbar import decode
import zxing
import math
from datetime import timedelta
from time import time, strftime, localtime
import atexit

from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.style.use('grayscale')

step = 2
color_threshold = 200


def time_past(start):
    end = time()
    elapsed = end-start
    return str(timedelta(seconds=elapsed))

# open image, send it to zbar and calculate
# perspective transformations
# return arrays: located qr-codes, may be incorretly rotated;
# perspective_transforms - matrix for cv2.warpPerspective;
# rects - rectangles in whitch qr-codes licated


def locate_qr(file_name):
    codes = decode(Image.open(file_name))
    # print(codes)
    located, perspective_transforms, rects = [], [], []
    for code in codes:
        if code.type == 'QRCODE':
            img = cv2.imread(file_name, 0)
            img = img[code.rect.top:code.rect.top+code.rect.height,
                      code.rect.left:code.rect.left+code.rect.width]

            pts1 = np.float32([[code.polygon[0][0]-code.rect.left, code.polygon[0][1]-code.rect.top],
                               [code.polygon[1][0]-code.rect.left,
                                   code.polygon[1][1]-code.rect.top],
                               [code.polygon[2][0]-code.rect.left,
                                   code.polygon[2][1]-code.rect.top],
                               [code.polygon[3][0]-code.rect.left, code.polygon[3][1]-code.rect.top]])
            pts2 = np.float32([[0, 0],
                               [code.rect.height, 0],
                               [code.rect.height, code.rect.height],
                               [0, code.rect.height]])
            perspective_transform = cv2.getPerspectiveTransform(pts1, pts2)

            located.append(cv2.warpPerspective(
                img, perspective_transform, (code.rect.height, code.rect.height)))
            perspective_transforms.append(perspective_transform)
            rects.append(code.rect)
    return located, perspective_transforms, rects


# return point which can be treated as corner without pattern and estimated block size;
# given point is outer corner of pattern
def find_patternless_corner(img, pattern_y, pattern_x, dy, dx):
    pattern_size = [1, 1, 3, 1, 1]
    pattern_color = [0, 255, 0, 255, 0]
    half_height = int(img.shape[1] / 3)
    size = [0, 0, 0, 0, 0]

    y, x = pattern_y, pattern_x
    for i in range(0, len(pattern_size)):
        if abs(img[y, x] - pattern_color[i]) > color_threshold:
            return [y, x]

        prev_y, prev_x = y, x
        while abs(img[prev_y, prev_x]-int(img[y, x])) < color_threshold:
            y += dy
            x += dx
            if abs(y - pattern_y) > half_height or abs(x - pattern_x) > half_height:
                return [y, x]

        size[i] += abs(prev_y - y) / pattern_size[i]

    average_size = sum(size) / len(size)
    if max(size) - min(size) > average_size:
        return [pattern_y, pattern_x]
    else:
        return average_size


# return correctly rotated QR-code and estimated block size
def rotate_qr(img):
    size = 0

    rotation_angle = None
    for y in range(step, img.shape[0], img.shape[0]-step*2):
        for x in range(step, img.shape[1], img.shape[1]-step*2):
            rows, cols = img.shape
            dy = int(np.sign(cols / 2 - y) * step)
            dx = int(np.sign(rows / 2 - x) * step)

            tmp = find_patternless_corner(img, y, x, dy, dx)

            if type(tmp) is list:
                if tmp[0] > cols / 2 and tmp[1] < rows / 2:
                    rotation_angle = 90
                elif tmp[0] < cols / 2 and tmp[1] < rows / 2:
                    rotation_angle = 180
                elif tmp[0] < cols / 2 and tmp[1] > rows / 2:
                    rotation_angle = 270
                else:
                    rotation_angle = 0
            else:
                size += tmp

    if 0 == size:
        raise RuntimeError(
            'Block size not found, mean all corners have not patterns')
    if rotation_angle is None:
        raise RuntimeError('Paternless corner not found')

    if rotation_angle != 0:
        rotation_matrix = cv2.getRotationMatrix2D(
            (cols/2, rows/2), rotation_angle, 1)
        img = cv2.warpAffine(img, rotation_matrix, (cols, rows))
    else:
        rotation_matrix = None

    size /= 3
    return size, img, rotation_matrix


# return width and height of block
# img is correctly rotated code
# est_size is estimated block size
def find_block_size(img, est_size):
    pattern_corner = [int(est_size * 7), int(est_size * 7)]

    color = int(img[pattern_corner[0], pattern_corner[1]])
    cnt = 0
    for y in range(pattern_corner[0], img.shape[0] - pattern_corner[0], step):
        if abs(img[y, pattern_corner[1]] - color) > color_threshold:
            cnt += 1
            color = int(img[y, pattern_corner[1]])
    size = [img.shape[0] / (cnt + 13), 0]

    color = int(img[pattern_corner[0], pattern_corner[1]])
    cnt = 0
    for x in range(pattern_corner[1], img.shape[1] - pattern_corner[1], step):
        if abs(img[pattern_corner[0], x] - color) > color_threshold:
            cnt += 1
            color = int(img[pattern_corner[0], x])

    size[1] = img.shape[1] / (cnt + 13)

    if img.shape[0] == img.shape[1] and size[0] != size[1]:
        size[0] = size[1] = (size[0] + size[1]) / 2

    return size


# uses kmeans to classify colors
# return lables - lables for each block of qr-code
#        lables_to_bits - array, indexes - lables, values - 3bit arrays,
#                         each can be treated as flag for having
#                         corresponding R, G or B color on this block
def cluster_colors(color_img, size, n_clusters):
    X = []
    x = size[1] / 2
    while(x < color_img.shape[1]):
        y = size[0] / 2
        while(y < color_img.shape[0]):
            X.append(color_img[int(y), int(x)])
            y += size[0]
        x += size[1]

    X = np.array(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    lables = kmeans.predict(X)

    thresholds = [0, 0, 0]
    for color in range(0, len(kmeans.cluster_centers_[0])):
        for cluster_center in kmeans.cluster_centers_:
            thresholds[color] += cluster_center[color]
        thresholds[color] /= len(kmeans.cluster_centers_)

    lables_to_bits = []

    for cluster_center in kmeans.cluster_centers_:
        tmp = []
        for color in range(0, len(cluster_center)):
            if cluster_center[color] > thresholds[color]:
                tmp.append(1)
            else:
                tmp.append(0)
        lables_to_bits.append(tmp)

    return lables, lables_to_bits


# make image of qr-code for each dimension of lables_to_bit[0]
# and send it to zbar
# return data from this codes
def recover_data_from_lables(lables, lables_to_bits, qr_shape, qr_margin):
    qr = Image.new('1', qr_shape, 1)
    pix = qr.load()
    reader = zxing.BarCodeReader()
    data = []

    for color in range(0, len(lables_to_bits[0])):
        i = 0
        for x in range(qr_margin[0], qr_shape[0] - qr_margin[0]):
            for y in range(qr_margin[1], qr_shape[1] - qr_margin[1]):
                pix[y, x] = (lables_to_bits[lables[i]][color])
                i += 1

        #plt.subplot(122), plt.imshow(qr), plt.title('qr')
        zbar_code = decode(qr)

        if not zbar_code:
            qr.save('tmp.jpeg', 'jpeg')
            zxing_code = reader.decode("tmp.jpeg", True)

            if not zxing_code:
                continue
            else:
                # print(zxing_code)
                data.append(zxing_code.raw)
        else:
            # print(zbar_code)
            data.append(zbar_code[0].data)
    return data


def decode_color_qr(file_name, n_colors=8, block_size_divider=3):
    start = time()

    codes, perspective_transform, rect = locate_qr(file_name)
    data = []
    i = 0
    for code in codes:
        size, img, rotation_matrix = rotate_qr(code)

        size_true = find_block_size(img, size)
        size_true[0], size_true[1] = round(
            size_true[0] / block_size_divider), round(size_true[1] / block_size_divider)

        # print('img shape = ', img.shape, 'true size = ',
        #      img.shape[0] / (21 + 7 + 7))
        #print('block size est = ', size)
        #print('block size = ', size_true)

        color_img = cv2.imread(file_name)
        color_img = color_img[rect[i].top:rect[i].top+rect[i].height,
                              rect[i].left:rect[i].left+rect[i].width]
        color_img = cv2.warpPerspective(
            color_img, perspective_transform[i], (rect[i].height, rect[i].height))
        if rotation_matrix is not None:
            color_img = cv2.warpAffine(
                color_img, rotation_matrix, (color_img.shape[0], color_img.shape[1]))

        lables, lables_to_bits = cluster_colors(
            color_img, size_true, n_colors)

        qr_block_size = (1, 1)
        qr_margin = (int(qr_block_size[0] * 4),
                     int(qr_block_size[1] * 4))  # white area around qr-code
        qr_shape = (int(math.sqrt(len(lables)) + qr_margin[0] * 2),
                    int(math.sqrt(len(lables)) + qr_margin[1] * 2))

        data.append(recover_data_from_lables(
            lables, lables_to_bits, qr_shape, qr_margin))

        i += 1
    return data, time_past(start)


def hard_decode_colors(file_name, n_colors=8, block_size=(2, 2)):
    start = time()

    img = cv2.imread(file_name)
    #plt.subplot(121), plt.imshow(img), plt.title('img')

    lables, lables_to_bits = cluster_colors(img, block_size, n_colors)

    #print('shape = ', img.shape)
    #print('step = ', block_size)

    qr_block_size = (1, 1)
    qr_margin = (block_size[0] * 4, block_size[1]
                 * 4)  # white area around qr-code
    qr_shape = (int(math.sqrt(len(lables)) + qr_margin[0] * 2),
                int(math.sqrt(len(lables)) + qr_margin[1] * 2))

    data = recover_data_from_lables(
        lables, lables_to_bits, qr_shape, qr_margin)

    return data, time_past(start)
