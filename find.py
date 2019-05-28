import cv2
import numpy as np
from matplotlib import pyplot as plt
from pyzbar.pyzbar import decode
from PIL import Image
from sklearn.cluster import KMeans

step = 2
color_threshold = 200


def locate_qr(file_name):
    codes = decode(Image.open(file_name))
    print(codes)
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
    half_height = img.shape[1] / 2
    size = 0

    for i in range(0, len(pattern_size)):
        if abs(img[pattern_y, pattern_x] - pattern_color[i]) > color_threshold:
            return [pattern_y, pattern_x]

        prev_y, prev_x = pattern_y, pattern_x
        while abs(img[prev_y, prev_x]-int(img[pattern_y, pattern_x])) < color_threshold:
            pattern_y += dy
            pattern_x += dx
            if abs(pattern_y - prev_y) > half_height or abs(pattern_x - prev_x) > half_height:
                return [prev_y, prev_x]

        size += abs(prev_y - pattern_y) / pattern_size[i]
    size /= len(pattern_size)
    return size


# return correctly rotated QR-code and estimated block size
def rotate_qr(img):
    size = 0

    rotation_angle = 0
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
                size += tmp

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
    return size


def recover_colors(color_img, size, n_clusters):
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

    color_to_bits = []

    for cluster_center in kmeans.cluster_centers_:
        tmp = []
        for color in range(0, len(cluster_center)):
            if cluster_center[color] > thresholds[color]:
                tmp.append(1)
            else:
                tmp.append(0)
        color_to_bits.append(tmp)

    return color_to_bits, lables


def lables_to_data(lables, lables_to_bits, img_shape):
    img = Image.new('1', img_shape)
    data = []

    for color in range(0, len(lables_to_bits[0])):
        for x in range(0, img_shape[0]):
            for y in range(0, img_shape[1]):
                img[x, y] = lables_to_bits[lables[(x+1)*y]][color]
        code = decode(img)
        data.append(code.data)
        print(code.data)


def prepare_img(file_name, n_clusters):
    codes, perspective_transform, rect = locate_qr(file_name)
    i = 0
    for code in codes:
        size, img, rotation_matrix = rotate_qr(code)
        size_true = find_block_size(img, size)

        color_img = cv2.imread(file_name)
        color_img = color_img[rect[i].top:rect[i].top+rect[i].height,
                              rect[i].left:rect[i].left+rect[i].width]
        color_img = cv2.warpPerspective(
            color_img, perspective_transform[i], (rect[i].height, rect[i].height))

        if rotation_matrix is not None:
            color_img = cv2.warpAffine(
                color_img, rotation_matrix, (color_img.shape[0], color_img.shape[1]))

        lables_to_bits, lables = recover_colors(
            color_img, size_true, n_clusters)
        lables_to_data(lables, lables_to_bits,
                       (color_img.shape[0], color_img.shape[1]))

        i += 1
    return imgs, sizes
