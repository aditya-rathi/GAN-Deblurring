import numpy as np
from tensorflow.keras.preprocessing.image import load_img
import glob
import time

TRAIN_DIR = "train\\"
TEST_DIR = "test\\"

TRAIN_FOLDERS = glob.glob(TRAIN_DIR +"*")
TEST_FOLDERS  = glob.glob(TEST_DIR +"*")

TRAIN_BLUR = [i + "\\blur\\" for i in TRAIN_FOLDERS] # Each blur folder in the trainset
TRAIN_BLUR_GAMMA = [i + "\\blur_gamma\\" for i in TRAIN_FOLDERS] # each blur gamma folder in the trainset
TRAIN_SHARP = [i + "\\sharp\\" for i in TRAIN_FOLDERS] # each sharp folder in the trainset

TEST_BLUR = [i + "\\blur\\" for i in TEST_FOLDERS] # Each blur folder in the testset
TEST_BLUR_GAMMA = [i + "\\blur_gamma\\" for i in TEST_FOLDERS] # each blur gamma folder in the testset
TEST_SHARP = [i + "\\sharp\\" for i in TEST_FOLDERS] # each sharp folder in the testset

# Concatenate all blur, blur gamma and sharp images in the same order, preserving the sample -> target match.

def read_images(collection, image_size = (256, 256)):
    
    all_images = []

    # Read each image in a single folder before moving to the next
    for folder in collection:
        
    # Get the paths of all images in the folder
        paths = glob.glob(folder + "*.png")

        for img in paths:
            image = load_img(img, target_size=image_size)
            image = np.array(image)/255.0
            all_images.append(image)

    return np.array(all_images)


if __name__ == "__main__":

    start = time.perf_counter()

    train_blur = read_images(TRAIN_BLUR)
    train_blur_gamma = read_images(TRAIN_BLUR_GAMMA)
    train_sharp = read_images(TRAIN_SHARP)
    test_blur = read_images(TEST_BLUR)
    test_blur_gamma = read_images(TEST_BLUR_GAMMA)
    test_sharp = read_images(TEST_SHARP)

    # save train set
    np.save("train_blur", train_blur)
    np.save("train_blur_gamma", train_blur_gamma)
    np.save("train_sharp", train_sharp)

    # save test set
    np.save("test_blur", test_blur)
    np.save("test_blur_gamma", test_blur_gamma)
    np.save("test_sharp", test_sharp)

    print(time.perf_counter() - start)

    print(train_blur.shape)