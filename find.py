import cv2
import numpy as np
from matplotlib import pyplot as plt
from pyzbar.pyzbar import decode
from PIL import Image


def locate_qrcode(file_name):
    codes = decode(Image.open(file_name))    
    decoded = []
    for code in codes:
        if code.type == 'QRCODE':
            print(code)
            img = cv2.imread(file_name)            
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
            decoded.append(cv2.warpPerspective(img,M,(code.rect.height,code.rect.height)))
            
            #plt.subplot(121),plt.imshow(img),plt.title('Input')
            #plt.subplot(122),plt.imshow(decoded[len(decoded)-1]),plt.title('Output')
            #plt.show()
    
    #cv2.imwrite("alpha.png", decoded[1])
    return decoded
