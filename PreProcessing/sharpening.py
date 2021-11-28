import numpy as np
import cv2
import os

list_blur = os.listdir('blur')
list_gblur = os.listdir('blur_gamma')

for k in [1,3,5]:
    for i in list_gblur:
        kernel = np.array([[0,-k,0],[-k,4*k+1,-k],[0,-k,0]])
        img = cv2.imread('blur_gamma/'+i)
        out = cv2.filter2D(img,-1,kernel)
        #cv2.imshow('nlm',out)
        #cv2.waitKey(3)
        cv2.imwrite('gblur_sharp/k'+str(k)+'/'+i,out)