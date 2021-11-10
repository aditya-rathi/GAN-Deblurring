import tensorflow as tf
import tensorflow.keras.layers as tfl

class ReflectionPadding2D(tfl.Layer):
    """
    Reflection padding layer.
    """
    def __init__(self, padding, **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [tfl.InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        shape = (
            input_shape[0],
            input_shape[1] + 2 * self.padding[0],
            input_shape[2] + 2 * self.padding[1],
            input_shape[3]
        )
        return shape

    def call(self, x, mask=None):
        width_pad, height_pad = self.padding
        return tf.pad(
            x,
            [[0, 0], [height_pad, height_pad], [width_pad, width_pad], [0, 0]],
            'REFLECT'
        )

def ResNet(input, filters, kernel_size=(3,3), stride=1, use_dropout=False):

    """
    One ResNet block. 

    Input: input from the previous layer.
    filters: number of channels to keep (256).
    kernel_size = sliding window size.
    stride: Sliding window stride to take.
    use_dropout: Whether to use dropout.
    """

    x = ReflectionPadding2D((1,1))(input)
    x = tfl.Conv2D(filtes=filters, kernel_size=kernel_size, strides=(stride,stride))(x)
    x = tfl.BatchNormalization()(x)
    x = tfl.ReLU()(x)

    if use_dropout:
        x = tfl.Dropout(0.4)(x) # Hyperparam
    
    x = ReflectionPadding2D((1,1))(x)
    x = tfl.Conv2D(filtes=filters, kernel_size=kernel_size, strides=(stride,stride))(x)
    x = tfl.BatchNormalization()(x)
    
    x = tfl.Add()([input, x]) # skip connection

    x = tfl.ReLU()(x) # Check this line. In ResNet but not in their code?

    return x