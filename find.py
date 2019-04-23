import cv2
import numpy as np
from matplotlib import pyplot as plt
from pyzbar.pyzbar import decode
from PIL import Image

step = 1
color_threshold = 200

def locate_qr(file_name):
    codes = decode(Image.open(file_name))
    located, perspective_transforms, rects = [], [], []
    for code in codes:
        if code.type == 'QRCODE':
            #print(code)
            img = cv2.imread(file_name,0)            
            img = img[code.rect.top:code.rect.top+code.rect.height,
                      code.rect.left:code.rect.left+code.rect.width]
                        
            pts1 = np.float32([[code.polygon[0][0]-code.rect.left, code.polygon[0][1]-code.rect.top],
                               [code.polygon[1][0]-code.rect.left, code.polygon[1][1]-code.rect.top],
                               [code.polygon[2][0]-code.rect.left, code.polygon[2][1]-code.rect.top],
                               [code.polygon[3][0]-code.rect.left, code.polygon[3][1]-code.rect.top]])
            pts2 = np.float32([[0,0],
                               [code.rect.height,0],
                               [code.rect.height,code.rect.height],
                               [0,code.rect.height]])
            
            perspective_transform = cv2.getPerspectiveTransform(pts1,pts2)            
            located.append(cv2.warpPerspective(img,perspective_transform, (code.rect.height,code.rect.height)))
            perspective_transforms.append(perspective_transform)
            rects.append(code.rect)
            
            #plt.subplot(121),plt.imshow(img),plt.title('Input')
            #plt.subplot(122),plt.imshow(located[len(located)-1]),plt.title('Output')
            #plt.show()
    return located, perspective_transforms, rects


# return point which can be treated as corner without pattern and estimated block size;
# given point is outer corner of pattern
def find_patternless_corner(img, pattern_y, pattern_x, dy, dx):
    pattern_size = [1,1,3,1,1]
    pattern_color = [0,255,0,255,0]
    half_height = img.shape[1] / 2
    size = 0
    
    for i in range(0,len(pattern_size)):
        if abs(img[pattern_y,pattern_x] - pattern_color[i]) > color_threshold:
            return [pattern_y, pattern_x]
        
        prev_y, prev_x = pattern_y, pattern_x
        while abs(img[prev_y,prev_x]-int(img[pattern_y,pattern_x])) < color_threshold:
            pattern_y += dy
            pattern_x += dx            
            if abs(pattern_y - prev_y) > half_height or abs(pattern_x - prev_x) > half_height:
                return [prev_y, prev_x]
        
        size += abs(prev_y - pattern_y) / pattern_size[i]
    size /= len(pattern_size)
    return size


# return correctly rotated QR-code and estimated block size
def rotate_qr(img):
    #plt.imshow(img),plt.title('Input')
    size = 0;

    for y in range(step,img.shape[0],img.shape[0]-step*2):
        for x in range(step,img.shape[1],img.shape[1]-step*2):
            rows,cols = img.shape
            dy = int(np.sign(cols / 2 - y) * step)
            dx = int(np.sign(rows / 2 - x) * step)
            #print('\n', y, dy, '\n', x, dx, '\n', img[y,x])
        
            tmp = find_patternless_corner(img, y, x, dy, dx)
            if type(tmp) is list:
                #plt.subplot(121),plt.imshow(img),plt.title('Output')
                #print(tmp[0], tmp[1])
                rotation_angle = 0
                if tmp[0] > cols / 2 and tmp[1] < rows / 2:                    
                    rotation_angle = 90
                elif tmp[0] < cols / 2 and tmp[1] < rows / 2:
                    rotation_angle = 180
                elif tmp[0] < cols / 2 and tmp[1] > rows / 2:
                    rotation_angle = 270
            else:
                size += tmp
                
    if rotation_angle != 0:
        rotation_matrix = cv2.getRotationMatrix2D((cols/2,rows/2),rotation_angle,1)
        img = cv2.warpAffine(img,rotation_matrix,(cols,rows))
    else:
        rotation_matrix = None    
    size /= 3
    return size, img, rotation_matrix


#return width and height of block
#img is correctly rotated code 
#est_size is estimated blick size
def find_block_size(img, est_size):
    pattern_corner = [int(est_size * 7), int(est_size * 7)]    
    color = int(img[pattern_corner[0], pattern_corner[1]])
    cnt = 0;    
    for y in range(pattern_corner[0], img.shape[0] - pattern_corner[0], step):
        #print('y=', y, 'x=', pattern_corner[1], 'color=', color, 'cur_color=', img[y, pattern_corner[1]])        
        if abs(img[y, pattern_corner[1]] - color) > color_threshold:            
            cnt += 1            
            color = int(img[y, pattern_corner[1]])
            #print('find, color=', color)
    #print('size_y=', img.shape[0] / (cnt + 13), 'cnt=', cnt)
    size = [img.shape[0] / (cnt + 13),0]
    
    color = int(img[pattern_corner[0], pattern_corner[1]])
    cnt = 0;
    for x in range(pattern_corner[1], img.shape[1] - pattern_corner[1], step):
        #print('y=', pattern_corner[0], 'x=', x, 'color=', color, 'cur_color=', img[pattern_corner[0], x])
        #cv2.circle(img,(pattern_corner[0], x), 2, (125,0,0), -1)
        if abs(img[pattern_corner[0], x] - color) > color_threshold:
            cnt += 1
            color = int(img[pattern_corner[0], x])
            #print('find, color=', color)    
    #print('size_x=', img.shape[1] / (cnt + 13), 'cnt=', cnt)
    #plt.subplot(122),plt.imshow(img),plt.title('Output')
    size[1] = img.shape[1] / (cnt + 13)
    return size


def prepare_colors(color_img, size):    
    X = []
    print(len(X), size)
    x = size[1] / 2
    while(x < color_img.shape[1]):
        y = size[0] / 2
        while(y < color_img.shape[0]):
            X.append(color_img[int(y),int(x)])
            y += size[0]
        x += size[1]
        
    X = np.array(X)
    print(X.shape, X)


def prepare_img(file_name):
    codes, perspective_transform, rect = locate_qr(file_name)
    sizes, imgs = [], []
    i = 0;
    for code in codes:
        size, img, rotation_matrix = rotate_qr(code)
        size_true = find_block_size(img, size)
        #print(size, size_new)
        plt.subplot(121),plt.imshow(img),plt.title('Input')
                
        color_img = cv2.imread(file_name)
        color_img = color_img[rect[i].top:rect[i].top+rect[i].height,
                              rect[i].left:rect[i].left+rect[i].width]
        color_img = cv2.warpPerspective(color_img,perspective_transform[i], (rect[i].height,rect[i].height))
        if rotation_matrix.all() != None:
            color_img = cv2.warpAffine(color_img,rotation_matrix,(color_img.shape[0],color_img.shape[1]))
            
        plt.subplot(122),plt.imshow(color_img),plt.title('Output')
        prepare_colors(color_img, size_true)
            
        imgs.append(color_img)
        sizes.append(size_true)
        
        i+=1
    return imgs, sizes    
