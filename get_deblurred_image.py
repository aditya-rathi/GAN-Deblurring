import numpy as np
from PIL import Image
import click
import os
import cv2

from generator import generator_model

def create_sharpening_kernel():
    kernel = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])
    return kernel

def apply_filter(img, kernel):
    return cv2.filter2D(img, -1, kernel)

def load_image(path):
    img = Image.open(path)
    # img = cv2.imread(path)
    # img_shape = img.shape
    # img = cv2.resize(img, (128,128))
    return img

def gamma_correction(img, gamma=1):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def increase_brightness(img, value=50):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def preprocess_image(img):
    img = img.resize((128,128))
    img = np.array(img)   
    # img = img/255
    return img

def deprocess_image(img):
    # img = img* 255
    return img.astype(np.uint8)

def deblur(weight_path, input_dir, output_dir):
    g = generator_model()
    g.load_weights(weight_path)
    for image_name in os.listdir(input_dir):
        l_img = load_image(os.path.join(input_dir, image_name))
        
        image = np.array([preprocess_image(l_img)])
        x_test = image
        generated_images = g.predict(x=x_test)
        generated = np.array([deprocess_image(img) for img in generated_images])
        x_test = deprocess_image(x_test)
        for i in range(generated_images.shape[0]):
            x = x_test[i, :, :, :]
            img = generated[i, :, :, :]
            output = np.concatenate((x, img), axis=1)
            im = Image.fromarray(output.astype(np.uint8))
            im = im.resize((1280, 720))
            im.save(os.path.join(output_dir, image_name))
            


weight_path = "D:\\shrey\\Documents\\1stsem\\cv\\generator.h5"
input_dir = "D:\\shrey\\Documents\\1stsem\\cv\\train\\GOPR0372_07_00\\blur\\"
output_dir = "D:\\shrey\\Documents\\1stsem\\cv\\new2"
deblur(weight_path, input_dir, output_dir)

# Training paths -
# GOPR0372_07_00
# GOPR0884_11_00
# GOPR0477_11_00
# GOPR0384_11_04
# GOPR0371_11_01
# GOPR0857_11_00
