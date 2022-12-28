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
from flexflow.keras.layers import Input, Flatten, Dense, Activation, Conv2D, MaxPooling2D, Concatenate, concatenate, Dropout, BatchNormalization, Add
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

def block1(x, filters, kernel_size=(3,3), stride=(1,1), conv_shortcut=True):
    if conv_shortcut is True:
        shortcut = Conv2D(4 * filters, kernel_size=(1,1), strides=stride)(x)
        # shortcut = BatchNormalization(axis=1)(shortcut)
    else:
        shortcut = x

    x = Conv2D(filters, kernel_size=(1,1), strides=stride)(x)
    # x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size=kernel_size, padding='same')(x)
    # x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)

    x = Conv2D(4 * filters, kernel_size=(1,1))(x)
    # x = BatchNormalization(axis=1)(x)

    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    return x

def stack1(x, filters, blocks, stride1=(2,2)):
    """A set of stacked residual blocks.
    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.
    # Returns
        Output tensor for the stacked blocks.
    """
    x = block1(x, filters, stride=stride1)
    for i in range(2, blocks + 1):
        x = block1(x, filters, conv_shortcut=False)
    return x


def top_level_task():
  num_samples = 96*5

  (x_train, y_train), (x_test, y_test) = cifar10.load_data(num_samples)

  full_input_np = np.zeros((num_samples, 3, 224, 224), dtype=np.float32)
  for i in range(0, num_samples):
    image = x_train[i, :, :, :]
    image = image.transpose(1, 2, 0)
    pil_image = Image.fromarray(image)
    pil_image = pil_image.resize((224,224), Image.NEAREST)
    image = np.array(pil_image, dtype=np.float32)
    image = image.transpose(2, 0, 1)
    full_input_np[i, :, :, :] = image
    if (i == 0):
      print(image)

  full_input_np /= 255
  y_train = y_train.astype('int32')
  full_label_np = y_train

  input_tensor = Input(shape=(3, 224, 224), dtype="float32")

  output = Conv2D(64, kernel_size=(7,7), strides=(2,2), padding=(3,3))(input_tensor)
  output = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding=(1,1))(input_tensor)

  output = stack1(input_tensor, 64, 3)
  output = stack1(output, 128, 4)
#   output = stack1(output, 256, 23)
  output = stack1(output, 256, 6)
  output = stack1(output, 512, 3, stride1=(1,1))

  output = MaxPooling2D(pool_size=(28,28), strides=(1,1), padding="valid")(output)

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
