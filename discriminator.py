import tensorflow as tf
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from generator import generator_model

# need to import generator functions

channelRate = 64
imgShape = (256, 256, 3)
patchShape = (channelRate, channelRate, 3)

ndf = 64
outputNC = 3
inputShapeD = (256, 256, outputNC)
# can potentially change the number of ResNet blocks
nBlocks = 9
kernel = (4, 4)

# discriminator architecture to determine if the input image is artificially
# created by the generator
def discriminator():
    nLayers, sigmoid = 3, False
    inputs = Input(shape=inputShapeD)

    x = Conv2D(filters = ndf, kernel_size = kernel, strides = 2, 
               padding = 'same')(inputs)
    # activation function between each CNN layer
    # handles dead neurons
    # alpha = 0.2
    x = LeakyReLU(0.2)(x)

    curr = 1
    prev = 1
    for n in range(nLayers):
        prev = curr
        curr = min(2**n, 8)
        x = Conv2D(filters = ndf * curr, kernel_size = kernel, strides = 2, 
                   padding = 'same')(x)
        # coordinate the update of multiple layers in the model before activation
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

    prev = curr
    curr = min(2**nLayers, 8)
    x = Conv2D(filters = ndf * curr, kernel_size = kernel, strides = 2, 
               padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filters = 1, kernel_size = kernel, strides = 1, 
               padding = 'same')(x)
    if (sigmoid): x = Activation('sigmoid')(x)
    #print(x.shape) # -> (8, 8, 1)
    # convert into 1D array (single feature vector)
    x = Flatten()(x)
    # matrix-vector multiplication with specified activation function
    x = Dense(1024, activation = 'tanh')(x)
    x = Dense(1, activation = 'sigmoid')(x)

    return Model(inputs = inputs, outputs = x, name = "Discriminator")


def generatorDiscriminator(generator, discriminator):
    # Multiple outputs
    inputs = Input(shape = imgShape)
    genImg = generator(inputs)
    outputs = discriminator(genImg)
    return Model(inputs = inputs, outputs = [genImg, outputs])

if __name__ == '__main__':
    # add generator function
    d = discriminator()
    print(d.summary())
    model = generatorDiscriminator(generator_model(), discriminator())
    model.summary()
