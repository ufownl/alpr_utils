# Copyright (c) 2019, RangerUFO
#
# This file is part of alpr_utils.
#
# alpr_utils is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# alpr_utils is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with alpr_utils.  If not, see <https://www.gnu.org/licenses/>.


import mxnet as mx


def add_layers(block, dims):
    block.add(
        mx.gluon.nn.BatchNorm(scale=False, center=False),
        mx.gluon.nn.Conv2D(dims // 8, 7, 2, 3),
        mx.gluon.nn.BatchNorm(),
        mx.gluon.nn.Activation("relu"),
        mx.gluon.model_zoo.vision.BasicBlockV2(dims // 8, 1),
        mx.gluon.model_zoo.vision.BasicBlockV2(dims // 8, 1),
        mx.gluon.model_zoo.vision.BasicBlockV2(dims // 4, 2, True),
        mx.gluon.model_zoo.vision.BasicBlockV2(dims // 4, 1),
        mx.gluon.model_zoo.vision.BasicBlockV2(dims // 2, 2, True),
        mx.gluon.model_zoo.vision.BasicBlockV2(dims // 2, 1),
        mx.gluon.model_zoo.vision.BasicBlockV2(dims, 2, True),
        mx.gluon.model_zoo.vision.BasicBlockV2(dims, 1),
        mx.gluon.nn.BatchNorm(),
        mx.gluon.nn.Activation("relu"),
    )


if __name__ == "__main__":
    block = mx.gluon.nn.Sequential()
    add_layers(block, 256)
    block.initialize(mx.init.Xavier())
    print(block)
    print(block(mx.nd.random_uniform(shape=(4, 3, 256, 256))))
    print(block(mx.nd.random_uniform(shape=(4, 3, 48, 144))))
