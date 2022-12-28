# Copyright 2020 Stanford University, Los Alamos National Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from flexflow.keras.models import Model, Sequential
from flexflow.keras.layers import Input, Flatten, Dense, Activation, Conv2D, MaxPooling2D, Concatenate, concatenate, Dropout, BatchNormalization, Add, AveragePooling2D
import flexflow.keras.optimizers
from flexflow.keras.datasets import mnist
from flexflow.keras.datasets import cifar10
from flexflow.keras import losses
from flexflow.keras import metrics

import flexflow.core as ff
import numpy as np
import argparse
import gc

from PIL import Image

def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1)):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    bn_axis = 1
    x = Conv2D(
        filters, kernel_size=(num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False)(x)
    # x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    return x

def top_level_task():
  num_samples = 96*5

  (x_train, y_train), (x_test, y_test) = cifar10.load_data(num_samples)

  full_input_np = np.zeros((num_samples, 3, 299, 299), dtype=np.float32)
  for i in range(0, num_samples):
    image = x_train[i, :, :, :]
    image = image.transpose(1, 2, 0)
    pil_image = Image.fromarray(image)
    pil_image = pil_image.resize((299,299), Image.NEAREST)
    image = np.array(pil_image, dtype=np.float32)
    image = image.transpose(2, 0, 1)
    full_input_np[i, :, :, :] = image
    if (i == 0):
      print(image)

  full_input_np /= 255
  y_train = y_train.astype('int32')
  full_label_np = y_train

  input_tensor = Input(shape=(3, 299, 299), dtype="float32")

  x = conv2d_bn(input_tensor, 32, 3, 3, strides=(2, 2), padding='valid')
  x = conv2d_bn(x, 32, 3, 3, padding='valid')
  x = conv2d_bn(x, 64, 3, 3)
  x = MaxPooling2D((3, 3), strides=(2, 2))(x)

  x = conv2d_bn(x, 80, 1, 1, padding='valid')
  x = conv2d_bn(x, 192, 3, 3, padding='valid')
  x = MaxPooling2D((3, 3), strides=(2, 2))(x)

  branch1x1 = conv2d_bn(x, 64, 1, 1)

  branch5x5 = conv2d_bn(x, 48, 1, 1)
  branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

  branch3x3dbl = conv2d_bn(x, 64, 1, 1)
  branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
  branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

  branch_pool = AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
  branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
  x = Concatenate(axis=1)(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool])

  # mixed 1: 35 x 35 x 288
  branch1x1 = conv2d_bn(x, 64, 1, 1)

  branch5x5 = conv2d_bn(x, 48, 1, 1)
  branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

  branch3x3dbl = conv2d_bn(x, 64, 1, 1)
  branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
  branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

  branch_pool = AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
  branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
  x = Concatenate(axis=1)(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool])

  # mixed 2: 35 x 35 x 288
  branch1x1 = conv2d_bn(x, 64, 1, 1)

  branch5x5 = conv2d_bn(x, 48, 1, 1)
  branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

  branch3x3dbl = conv2d_bn(x, 64, 1, 1)
  branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
  branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

  branch_pool = AveragePooling2D((3, 3),
                                          strides=(1, 1))(x)
  branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
  x = Concatenate(axis=1)(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool])

  # mixed 3: 17 x 17 x 768
  branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

  branch3x3dbl = conv2d_bn(x, 64, 1, 1)
  branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
  branch3x3dbl = conv2d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

  branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
  x = Concatenate(axis=1)(
        [branch3x3, branch3x3dbl, branch_pool])

    # mixed 4: 17 x 17 x 768
  branch1x1 = conv2d_bn(x, 192, 1, 1)

  branch7x7 = conv2d_bn(x, 128, 1, 1)
  branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
  branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

  branch7x7dbl = conv2d_bn(x, 128, 1, 1)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

  branch_pool = AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
  branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
  x = Concatenate(axis=1)(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool])

  # mixed 5, 6: 17 x 17 x 768
  for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = Concatenate(axis=1)(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool])

    # mixed 7: 17 x 17 x 768
  branch1x1 = conv2d_bn(x, 192, 1, 1)

  branch7x7 = conv2d_bn(x, 192, 1, 1)
  branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
  branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

  branch7x7dbl = conv2d_bn(x, 192, 1, 1)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
  branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

  branch_pool = AveragePooling2D((3, 3),
                                          strides=(1, 1))(x)
  branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
  x = Concatenate(axis=1)(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool])

  # mixed 8: 8 x 8 x 1280
  branch3x3 = conv2d_bn(x, 192, 1, 1)
  branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='valid')

  branch7x7x3 = conv2d_bn(x, 192, 1, 1)
  branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
  branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
  branch7x7x3 = conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

  branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
  x = Concatenate(axis=1)(
        [branch3x3, branch7x7x3, branch_pool])

    # mixed 9: 8 x 8 x 2048
  for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = Concatenate(axis=1)(
            [branch3x3_1, branch3x3_2])

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = Concatenate(axis=1)(
            [branch3x3dbl_1, branch3x3dbl_2], axis=1)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = Concatenate(axis=1)(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool])


  output = Flatten()(output)
  output = Dense(1000)(output)
  output = Activation("softmax")(output)

  model = Model(input_tensor, output)
  print(model.summary())

  opt = flexflow.keras.optimizers.SGD(learning_rate=0.000)
  model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy', 'sparse_categorical_crossentropy'])
  print(model.summary())

  model.fit(full_input_np, full_label_np, epochs=1, callbacks=[])

if __name__ == "__main__":
  print("Functional API, cifar10 alexnet")
  top_level_task()
  gc.collect()
