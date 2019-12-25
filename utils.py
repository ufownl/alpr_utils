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


import cv2
import json
import math
import random
import numpy as np
import mxnet as mx

def points_matrix(pts):
    return np.matrix(np.concatenate((pts, np.ones((1, pts.shape[1]))), 0))

def rect_matrix(tlx, tly, brx, bry):
    return np.matrix([
        [tlx, brx, brx, tlx],
        [tly, tly, bry, bry],
        [1.0, 1.0, 1.0, 1.0]
    ])

def transform_matrix(pts, t_pts):
    return cv2.getPerspectiveTransform(np.float32(pts[:2, :].T), np.float32(t_pts[:2, :].T))

def rotate_matrix(width, height, angles=np.zeros(3), zcop=1000.0, dpp=1000.0):
    rads = np.deg2rad(angles)
    rx = np.matrix([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(rads[0]), math.sin(rads[0])],
        [0.0, -math.sin(rads[0]), math.cos(rads[0])]
    ])
    ry = np.matrix([
        [math.cos(rads[1]), 0.0, -math.sin(rads[1])],
        [0.0, 1.0, 0.0],
        [math.sin(rads[1]), 0.0, math.cos(rads[1])]
    ])
    rz = np.matrix([
        [math.cos(rads[2]), math.sin(rads[2]), 0.0],
        [-math.sin(rads[2]), math.cos(rads[2]), 0.0],
        [0.0, 0.0, 1.0]
    ])
    r = rx * ry * rz
    hxy = np.matrix([
        [0.0, 0.0, width, width],
        [0.0, height, 0.0, height],
        [1.0, 1.0, 1.0, 1.0]
    ])
    xyz = np.matrix([
        [0.0, 0.0, width, width],
        [0.0, height, 0.0, height],
        [0.0, 0.0, 0.0, 0.0]
    ])
    half = np.matrix([[width], [height], [0.0]]) / 2.0
    xyz = r * (xyz - half) - np.matrix([[0.0], [0.0], [zcop]])
    xyz = np.concatenate((xyz, np.ones((1, 4))), 0)
    p = np.matrix([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0 / dpp, 0.0]
    ])
    t_hxy = p * xyz
    t_hxy = t_hxy / t_hxy[2, :] + half
    return transform_matrix(hxy, t_hxy)

def project(img, pts, trans, dims):
    t_img = cv2.warpPerspective(img, trans, (dims, dims))
    t_pts = np.matmul(trans, points_matrix(pts))
    t_pts = t_pts / t_pts[2]
    return t_img, t_pts[:2]

def hsv_noise(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:, :, 0] = hsv[:, :, 0] * (0.8 + random.uniform(0.0, 0.2))
    hsv[:, :, 1] = hsv[:, :, 1] * (0.3 + random.uniform(0.0, 0.7))
    hsv[:, :, 2] = hsv[:, :, 2] * (0.2 + random.uniform(0.0, 0.8))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def brightness_noise(img, ratio=0.8):
    return np.clip(img * (1.0 + random.uniform(-ratio, ratio)), 0, 255)

def augment_sample(image, points, dims, flip_prob=0.5):
    image = image.astype("uint8").asnumpy()
    points = np.array(points).reshape((2, 4))
    points = points * np.array([[image.shape[1]], [image.shape[0]]])
    # random crop
    wh_ratio = random.uniform(2.0, 4.0)
    width = random.uniform(dims * 0.2, dims * 1.0)
    height = width / wh_ratio
    dx = random.uniform(0.0, dims - width)
    dy = random.uniform(0.0, dims - height)
    crop = transform_matrix(
        points_matrix(points),
        rect_matrix(dx, dy, dx + width, dy + height)
    )
    # random rotate
    max_angles = np.array([80.0, 80.0, 45.0])
    angles = np.random.rand(3) * max_angles
    if angles.sum() > 120:
        angles = (angles / angles.sum()) * (max_angles / max_angles.sum())
    rotate = rotate_matrix(dims, dims, angles)
    # apply projection
    image, points = project(image, points, np.matmul(rotate, crop), dims)
    # scale the coordinates of points to [0, 1]
    points = points / dims
    # random flip
    if random.random() < flip_prob:
        image = cv2.flip(image, 1)
        points[0] = 1 - points[0]
        points = points[..., [1, 0, 3, 2]]
    # color augment
    image = hsv_noise(image)
    # brightness augment
    image = brightness_noise(image)
    return mx.nd.array(image), np.asarray(points).reshape((-1,)).tolist()

def color_normalize(img):
    return mx.image.color_normalize(
        img.astype("float32") / 255,
        mean = mx.nd.array([0.485, 0.456, 0.406]),
        std = mx.nd.array([0.229, 0.224, 0.225])
    )

def point_in_polygon(x, y, pts):
    n = len(pts) // 2
    pts_x = [pts[i] for i in range(0, n)]
    pts_y = [pts[i] for i in range(n, len(pts))]
    if not min(pts_x) <= x <= max(pts_x) or not min(pts_y) <= y <= max(pts_y):
        return False
    res = False
    for i in range(n):
        j = n - 1 if i == 0 else i - 1
        if ((pts_y[i] > y) != (pts_y[j] > y)) and (x < (pts_x[j] - pts_x[i]) * (y - pts_y[i]) / (pts_y[j] - pts_y[i]) + pts_x[i]):
            res = not res
    return res

def object_label(points, dims, stride):
    scale = ((dims + 40.0) / 2.0) / stride
    size = dims // stride
    label = mx.nd.zeros((size, size, 9))
    for i in range(size):
        y = (i + 0.5) / size
        for j in range(size):
            x = (j + 0.5) / size
            if point_in_polygon(x, y, points):
                label[i, j, 0] = 1
                pts = mx.nd.array(points).reshape((2, -1))
                pts = pts * dims / stride
                pts = pts - mx.nd.array([[j + 0.5], [i + 0.5]])
                pts = pts / scale
                label[i, j, 1:] = pts.reshape((-1,))
    return label

def iou(tl1, br1, tl2, br2):
    wh1 = br1 - tl1
    wh2 = br2 - tl2
    assert((wh1 >= 0).sum() > 0 and (wh2 >= 0).sum() > 0)
    itl = mx.nd.concat(tl1.expand_dims(0), tl2.expand_dims(0), dim=0).max(axis=0)
    ibr = mx.nd.concat(br1.expand_dims(0), br2.expand_dims(0), dim=0).min(axis=0)
    iwh = mx.nd.relu(ibr - itl)
    ia = iwh.prod().asscalar()
    ua = wh1.prod().asscalar() + wh2.prod().asscalar() - ia
    return ia / ua

def plate_labels(image, probs, affines, dims, stride, threshold):
    wh = mx.nd.array([[image.shape[1]], [image.shape[0]]], ctx=affines.context)
    scale = ((dims + 40.0) / 2.0) / stride
    unit = mx.nd.array(
        [[-0.5, -0.5, 1], [0.5, -0.5, 1], [0.5, 0.5, 1], [-0.5, 0.5, 1]],
        ctx=affines.context
    ).T
    candidates = []
    for x, y in [(j, i) for i in range(probs.shape[0]) for j in range(probs.shape[1]) if probs[i, j] > threshold]:
        affine = mx.nd.concat(
            mx.nd.concat(
                mx.nd.relu(affines[y, x, 0]),
                affines[y, x, 1],
                affines[y, x, 2],
                dim=0
            ).expand_dims(0),
            mx.nd.concat(
                affines[y, x, 3],
                mx.nd.relu(affines[y, x, 4]),
                affines[y, x, 5],
                dim=0
            ).expand_dims(0),
            dim=0
        )
        pts = mx.nd.dot(affine, unit) * scale
        pts = pts + mx.nd.array([[x + 0.5], [y + 0.5]], ctx=pts.context)
        pts = pts * stride / wh
        candidates.append((pts, probs[y, x].asscalar()))
    candidates.sort(key=lambda x: x[1], reverse=True)
    labels = []
    for pts_c, prob_c in candidates:
        tl_c = pts_c.min(axis=1)
        br_c = pts_c.max(axis=1)
        overlap = False
        for pts_l, _ in labels:
            tl_l = pts_l.min(axis=1)
            br_l = pts_l.max(axis=1)
            if iou(tl_c, br_c, tl_l, br_l) > 0.1:
                overlap = True
                break
        if not overlap:
            labels.append((pts_c, prob_c))
    return labels

def reconstruct_plates(image, plate_pts, out_size=(240, 80)):
    wh = np.array([[image.shape[1]], [image.shape[0]]])
    plates = []
    for pts in plate_pts:
        pts = points_matrix(pts.asnumpy() * wh)
        t_pts = rect_matrix(0, 0, out_size[0], out_size[1])
        m = transform_matrix(pts, t_pts)
        plate = cv2.warpPerspective(image.astype("uint8").asnumpy(), m, out_size)
        plates.append(mx.nd.array(plate))
    return plates

def apply_plate(image, points, plate):
    image = image.astype("uint8").asnumpy()
    plate = plate.astype("uint8").asnumpy()
    points = np.array(points).reshape((2, 4))
    points = points * np.array([[image.shape[1]], [image.shape[0]]])
    pts = rect_matrix(0, 0, plate.shape[1], plate.shape[0])
    t_pts = points_matrix(points)
    m = transform_matrix(pts, t_pts)
    mask = np.ones_like(plate, dtype=np.uint8)
    mask = cv2.warpPerspective(mask, m, (image.shape[1], image.shape[0]))
    mask = (mask == 0).astype(np.uint8) * 255
    plate = cv2.warpPerspective(plate, m, (image.shape[1], image.shape[0]))
    return mx.nd.array(cv2.bitwise_or(cv2.bitwise_and(image, mask), plate))


class Vocabulary:
    def __init__(self, chars=None):
        if chars:
            self._chars = ["<PAD>", "<UNK>", "<GO>", "<EOS>"] + chars
            self._char_indices = dict((c, i) for i, c in enumerate(self._chars))
            self._indices_char = dict((i, c) for i, c in enumerate(self._chars))

    def size(self):
        return len(self._chars)

    def char2idx(self, ch):
        if ch not in self._char_indices:
            ch = "<UNK>"
        return self._char_indices[ch]

    def idx2char(self, idx):
        return self._indices_char[idx]

    def save(self, path):
        with open(path, "w") as f:
            f.write(json.dumps(self._chars))

    def load(self, path):
        with open(path, "r") as f:
            s = f.read()
        self._chars = json.loads(s)
        self._char_indices = dict((c, i) for i, c in enumerate(self._chars))
        self._indices_char = dict((i, c) for i, c in enumerate(self._chars))
