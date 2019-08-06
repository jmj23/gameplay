import keras.backend as K
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.initializers import glorot_uniform
from keras.layers import (ELU, Activation, Add, AveragePooling2D,
                          BatchNormalization, Conv2D, Dense, Flatten,
                          GlobalAveragePooling2D, Input, MaxPooling2D,
                          ZeroPadding2D, concatenate)
from keras.models import Model

K.set_image_data_format('channels_last')
K.set_learning_phase(1)

def BlockModel(input_shape, out_chan,filt_num=8, numBlocks=3):
    # input layer
    lay_input = Input(shape=input_shape, name='input_layer')

    # initial strided layer
    x = Conv2D(filt_num,(4,4),strides=(2,2),padding='valid')(lay_input)
    x = ELU(name='elu_start')(x)

    # contracting blocks
    for rr in range(1, numBlocks+1):
        x1 = Conv2D(filt_num*(2**(rr-1)), (1, 1), padding='same',
                    name='Conv1_{}'.format(rr))(x)
        # x1 = BatchNormalization()(x1)
        x1 = ELU(name='elu_x1_{}'.format(rr))(x1)
        x3 = Conv2D(filt_num*(2**(rr-1)), (3, 3), padding='same',
                    name='Conv3_{}'.format(rr))(x)
        # x3 = BatchNormalization()(x3)
        x3 = ELU(name='elu_x3_{}'.format(rr))(x3)
        x51 = Conv2D(filt_num*(2**(rr-1)), (3, 3), padding='same',
                     name='Conv51_{}'.format(rr))(x)
        # x51 = BatchNormalization()(x51)
        x51 = ELU(name='elu_x51_{}'.format(rr))(x51)
        x52 = Conv2D(filt_num*(2**(rr-1)), (3, 3), padding='same',
                     name='Conv52_{}'.format(rr))(x51)
        # x52 = BatchNormalization()(x52)
        x52 = ELU(name='elu_x52_{}'.format(rr))(x52)
        x = concatenate([x1, x3, x52], name='merge_{}'.format(rr))
        x = Conv2D(filt_num*(2**(rr-1)), (1, 1), padding='valid',
                   name='ConvAll_{}'.format(rr))(x)
        x = BatchNormalization()(x)
        x = ELU(name='elu_all_{}'.format(rr))(x)
        x = Conv2D(filt_num*(2**(rr-1)), (4, 4), padding='valid',
                   strides=(2, 2), name='DownSample_{}'.format(rr))(x)
        # x = BatchNormalization()(x)
        x = ELU(name='elu_downsample_{}'.format(rr))(x)
        x = Conv2D(filt_num*(2**(rr-1)), (3, 3), padding='same',
                   name='ConvClean_{}'.format(rr))(x)
        # x = BatchNormalization()(x)
        x = ELU(name='elu_clean_{}'.format(rr))(x)

    # average pooling
    x = Flatten()(x)
    # classifier
    lay_out = Dense(out_chan, activation='linear', name='output_layer')(x)

    return Model(lay_input, lay_out)

if __name__ == '__main__':
    testModel = BlockModel((256,256,1),4)
    testModel.summary()
