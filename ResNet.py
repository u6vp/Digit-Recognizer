# ブログから参照 (https://qiita.com/_odasi_/items/2d441e46643f73c0ca19)
from functools import reduce
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Activation, Add, concatenate, GlobalAveragePooling2D,BatchNormalization, Conv2D, Dense, Flatten, Input, MaxPooling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers

def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

def ResNetConv2D(*args, **kwargs):
    conv_kwargs = {
        'strides': (1, 1),
        'padding': 'same',
        'kernel_initializer': 'he_normal',
        'kernel_regularizer': l2(1.e-4)
    }
    conv_kwargs.update(kwargs)

    return Conv2D(*args, **conv_kwargs)

def bn_relu_conv(*args, **kwargs):
    return compose(
        BatchNormalization(),
        Activation('relu'),
        ResNetConv2D(*args, **kwargs))

def shortcut(x, residual):
    x_shape = K.int_shape(x)
    residual_shape = K.int_shape(residual)

    if x_shape == residual_shape:
        shortcut = x
    else:
        stride_w = int(round(x_shape[1] / residual_shape[1]))
        stride_h = int(round(x_shape[2] / residual_shape[2]))

        shortcut = Conv2D(filters=residual_shape[3],
                          kernel_size=(1, 1),
                          strides=(stride_w, stride_h),
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(1.e-4))(x)
    return Add()([shortcut, residual])

def basic_block(filters, first_strides, is_first_block_of_first_layer):
    def f(x):
        if is_first_block_of_first_layer:
            conv1 = ResNetConv2D(filters=filters, kernel_size=(3, 3))(x)
        else:
            conv1 = bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                 strides=first_strides)(x)

        conv2 = bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)

        return shortcut(x, conv2)

    return f

def bottleneck_block(filters, first_strides, is_first_block_of_first_layer):
    def f(x):
        if is_first_block_of_first_layer:
            conv1 = ResNetConv2D(filters=filters, kernel_size=(3, 3))(x)
        else:
            conv1 = bn_relu_conv(filters=filters, kernel_size=(1, 1),
                                 strides=first_strides)(x)

        conv2 = bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        conv3 = bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv2)

        return shortcut(x, conv3)

    return f

def residual_blocks(block_function, filters, repetitions, is_first_layer):
    def f(x):
        for i in range(repetitions):
            first_strides = (2, 2) if i == 0 and not is_first_layer else (1, 1)

            x = block_function(filters=filters, first_strides=first_strides,
                               is_first_block_of_first_layer=(i == 0 and is_first_layer))(x)
        return x

    return f

# ResNEt_build.py
class ResnetBuilder():
    @staticmethod
    def build(input_shape, num_outputs, block_type, repetitions):
        if block_type == 'basic':
            block_fn = basic_block
        elif block_type == 'bottleneck':
            block_fn = bottleneck_block

        input = Input(shape=input_shape)

        conv1 = compose(ResNetConv2D(filters=64, kernel_size=(7, 7), strides=(2, 2)),
                        BatchNormalization(),
                        Activation('relu'))(input)

        pool1 = MaxPooling2D(
            pool_size=(3, 3), strides=(2, 2), padding='same')(conv1)

        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
            block = residual_blocks(block_fn, filters=filters, repetitions=r,
                                    is_first_layer=(i == 0))(block)
            filters *= 2

        block = compose(BatchNormalization(),
                        Activation('relu'))(block)

        pool2 = GlobalAveragePooling2D()(block)

        fc1 = Dense(units=num_outputs,
                    kernel_initializer='he_normal',
                    activation='softmax')(pool2)

        return Model(inputs=input, outputs=fc1)

    @staticmethod
    def build_resnet_18(input_shape, num_outputs):
        return ResnetBuilder.build(
            input_shape, num_outputs, 'basic', [2, 2, 2, 2])

    @staticmethod
    def build_resnet_34(input_shape, num_outputs):
        return ResnetBuilder.build(
            input_shape, num_outputs, 'basic', [3, 4, 6, 3])

    @staticmethod
    def build_resnet_50(input_shape, num_outputs):
        return ResnetBuilder.build(
            input_shape, num_outputs, 'bottleneck', [3, 4, 6, 3])

    @staticmethod
    def build_resnet_101(input_shape, num_outputs):
        return ResnetBuilder.build(
            input_shape, num_outputs, 'bottleneck', [3, 4, 23, 3])

    @staticmethod
    def build_resnet_152(input_shape, num_outputs):
        return ResnetBuilder.build(
            input_shape, num_outputs, 'bottleneck', [3, 8, 36, 3])