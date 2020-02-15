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


import backbone
import mxnet as mx


class Detection(mx.gluon.nn.Block):
    def __init__(self, **kwargs):
        super(Detection, self).__init__(**kwargs)
        with self.name_scope():
            self._classify = mx.gluon.nn.Conv2D(2, 3, 1, 1)
            self._affine = mx.gluon.nn.Conv2D(6, 3, 1, 1)

    def forward(self, x):
        return mx.nd.concat(
            mx.nd.softmax(self._classify(x), axis=1),
            self._affine(x),
            dim=1
        )


class WpodNet(mx.gluon.nn.Block):
    def __init__(self, **kwargs):
        super(WpodNet, self).__init__(**kwargs)
        with self.name_scope():
            self._block = mx.gluon.nn.Sequential()
            backbone.add_layers(self._block, 256)
            self._block.add(Detection())

    def forward(self, x):
        return self._block(x).transpose((0, 2, 3, 1))


class LogLoss:
    def __init__(self, batch_axis=0):
        self._batch_axis = batch_axis

    def __call__(self, pred, label, epsilon=1e-8):
        loss = -label * mx.nd.log(mx.nd.clip(pred, epsilon, 1.0))
        return mx.nd.mean(loss, axis=self._batch_axis, exclude=True)


class WpodLoss:
    def __init__(self, batch_axis=0):
        self._log_loss = LogLoss(batch_axis)
        self._l1_loss = mx.gluon.loss.L1Loss(batch_axis=batch_axis)

    def __call__(self, pred, label, epsilon=1e-8):
        obj_pred = pred[:, :, :, 0]
        obj_label = label[:, :, :, 0]
        bg_pred = pred[:, :, :, 1]
        bg_label = 1 - obj_label
        classify_L = self._log_loss(obj_pred, obj_label, epsilon) + self._log_loss(bg_pred, bg_label, epsilon)
        affine = mx.nd.concat(
            mx.nd.concat(
                mx.nd.relu(pred[:, :, :, 2:3]),
                pred[:, :, :, 3:4],
                pred[:, :, :, 4:5],
                dim=3
            ).expand_dims(3),
            mx.nd.concat(
                pred[:, :, :, 5:6],
                mx.nd.relu(pred[:, :, :, 6:7]),
                pred[:, :, :, 7:8],
                dim=3
            ).expand_dims(3),
            dim=3
        )
        unit = mx.nd.array(
            [[-0.5, -0.5, 1], [0.5, -0.5, 1], [0.5, 0.5, 1], [-0.5, 0.5, 1]],
            ctx=pred.context
        ).T.reshape((1, 1, 1, 3, 4)).tile(pred.shape[:3] + (1, 1))
        points_pred = mx.nd.batch_dot(
            affine.reshape((-1, 2, 3)),
            unit.reshape((-1, 3, 4))
        ).reshape(pred.shape[:3] + (-1,))
        points_label = label[:, :, :, 1:]
        mask = obj_label.expand_dims(3)
        affine_L = self._l1_loss(points_pred * mask, points_label * mask)
        return classify_L + affine_L


if __name__ == "__main__":
    net = WpodNet()
    net.initialize(mx.init.Xavier())
    print(net)
    pred = net(mx.nd.random_uniform(shape=(4, 3, 256, 256)))
    print(pred)
    loss = WpodLoss()
    print(loss(pred, mx.nd.random_uniform(shape=(4, 16, 16, 9))))
