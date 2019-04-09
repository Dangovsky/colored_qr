import cv2
import numpy as np
from matplotlib import pyplot as plt
from pyzbar.pyzbar import decode
from PIL import Image

step = 2
color_threshold = 200

def locate_qrcode(file_name):
    codes = decode(Image.open(file_name))
    located = []
    for code in codes:
        if code.type == 'QRCODE':
            print(code)
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
            
            M = cv2.getPerspectiveTransform(pts1,pts2)
            located.append(cv2.warpPerspective(img,M,(code.rect.height,code.rect.height)))
            
            #plt.subplot(121),plt.imshow(img),plt.title('Input')
            #plt.subplot(122),plt.imshow(located[len(located)-1]),plt.title('Output')
            #plt.show()
    return located


# return block size if given point is outer corner of pattern, 
# and point otherwise (which can be treated as corner without pattern)
def find_block_size(img, pattern_y, pattern_x, dy, dx):
    pattern_size = [1,1,3,1,1]
    pattern_color = [0,255,0,255,0]
    half_height = img.shape[1] / 2
    size = 0
    
    for i in range(0,len(pattern_size)):
        if(abs(img[pattern_y,pattern_x] - pattern_color[i]) > color_threshold):
            return [pattern_y, pattern_x]
        
        prev_y, prev_x = pattern_y, pattern_x
        while(abs(img[prev_y,prev_x]-int(img[pattern_y,pattern_x])) < color_threshold):                
            pattern_y += dy
            pattern_x += dx            
            if (abs(pattern_y - prev_y) > half_height or abs(pattern_x - prev_x) > half_height):                    
                return [prev_y, prev_x]
        
        size += abs(prev_y - pattern_y) / pattern_size[i]
    size /= len(pattern_size)
    return int(np.ceil(size))


def rotate_and_find_block_size(img):
    #plt.imshow(img),plt.title('Input')
    size = 0;

    for y in range(step,img.shape[0],img.shape[0]-step*2):
        for x in range(step,img.shape[1],img.shape[1]-step*2):
            rows,cols = img.shape
            dy = int(np.sign(cols / 2 - y) * step)
            dx = int(np.sign(rows / 2 - x) * step)
            #print('\n', y, dy, '\n', x, dx, '\n', img[y,x])
        
            tmp = find_block_size(img, y, x, dy, dx)
            if type(tmp) is list:
                rotation_angle = 0
                if tmp[0] < cols / 2 and tmp[1] < rows / 2:
                    rotation_angle = 180
                elif tmp[0] > cols / 2 and tmp[1] < rows / 2:                    
                    rotation_angle = 90
                elif tmp[0] < cols / 2 and tmp[1] > rows / 2:
                    rotation_angle = 270
                
                if rotation_angle != 0:
                    M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
                    dst = cv2.warpAffine(img,M,(cols,rows))
            else:
                size += tmp
    size /= 3
    return size, img
    
    
def find_qrcode(file_name):
    codes = locate_qrcode(file_name)
    sizes, imgs = [], []
    for code in codes:
        size, img = rotate_and_find_block_size(code)
        plt.subplot(121),plt.imshow(cv2.imread(file_name)),plt.title('Input')
        plt.subplot(122),plt.imshow(img),plt.title('Output')
        print(size)
        sizes.append(size)
        imgs.append(img)
    return sizes, imgs
