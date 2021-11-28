import numpy as np
import cv2
import os

list_blur = os.listdir('blur')
list_gblur = os.listdir('blur_gamma')

for h in [3,5,7]:
    for i in list_blur:
        img = cv2.imread('blur/'+i)
        out = cv2.fastNlMeansDenoisingColored(img,h = h,hColor=10)
        #cv2.imshow('nlm',out)
        #cv2.waitKey(3)
        cv2.imwrite('blur_nlm/h'+str(h)+'/'+i,out)


