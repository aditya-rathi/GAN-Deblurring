import cv2
import os
# blur = os.listdir('blur')
blur = os.listdir('blur_gamma')
print(blur)
# for i in range(9):
#     for j in blur:
#         image = cv2.imread("blur/"+j)
#         gaussian_3 = cv2.GaussianBlur(image, (2*i + 1, 2*i + 1), 2.0)
#         unsharp_image = cv2.addWeighted(image, 2.0, gaussian_3, -1.0, 0)
#         cv2.imwrite("blur_res/" + str(j)+ str(i) + ".jpg", unsharp_image)
for j in blur:
    image = cv2.imread("blur_gamma/"+j)
    gaussian_3 = cv2.GaussianBlur(image, (19, 19), 2.0)
    unsharp_image = cv2.addWeighted(image, 2.0, gaussian_3, -1.0, 0)
    cv2.imwrite("blur_gamma_res/" + str(j) + ".jpg", unsharp_image)