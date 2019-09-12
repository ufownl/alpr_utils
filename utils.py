import cv2
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
    a = np.zeros((8, 9))
    for i in range(0, 4):
        x = pts[:, i].T
        tx = t_pts[:, i]
        a[i * 2, 3:6] = -tx[2] * x
        a[i * 2, 6:] = tx[1] * x
        a[i * 2 + 1, :3] = tx[2] * x
        a[i * 2 + 1, 6:] = -tx[0] * x
    u, s, v = np.linalg.svd(a)
    return v[-1, :].reshape((3, 3))

def perspective_transform_matrix(width, height, angles=np.zeros(3), zcop=1000.0, dpp=1000.0):
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
    xyz = r * (xyz - np.matrix([[width], [height], [0.0]]) / 2.0)
    xyz = xyz - np.matrix([[0.0], [0.0], [zcop]])
    xyz = np.concatenate((xyz, np.ones((1, 4))), 0)
    p = np.matrix([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0 / dpp, 0.0]
    ])
    t_hxy = p * xyz
    t_hxy = t_hxy / t_hxy[2, :] + np.matrix([[width], [height], [0.0]]) / 2.0
    return transform_matrix(hxy, t_hxy)

def project(img, pts, trans, dims):
    t_img = cv2.warpPerspective(img, trans, (dims, dims), flags=cv2.INTER_LINEAR)
    t_pts = np.matmul(trans, points_matrix(pts))
    t_pts = t_pts / t_pts[2]
    return t_img, t_pts[:2]

def hsv_transform(img, hsv_mod):
    img = img.astype(np.float32) / 255
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = img + hsv_mod
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)

def augment_sample(image, points, dims):
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
    rotate = perspective_transform_matrix(dims, dims, angles)
    # apply projection
    trans = np.matmul(rotate, crop)
    image, points = project(image, points, np.matmul(rotate, crop), dims)
    # color augment
    hsv_mod = (np.random.rand(3).astype(np.float32) - 0.5) * 0.3
    hsv_mod[0] *= 360
    image = hsv_transform(image, hsv_mod)
    return mx.nd.array(image), np.asarray(points / dims).reshape((-1,)).tolist()

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

def object_label(image, points, stride):
    height = image.shape[0] // stride
    width = image.shape[1] // stride
    label = mx.nd.zeros((height, width, 9))
    for i in range(height):
        y = (i + 0.5) / height
        for j in range(width):
            x = (j + 0.5) / width
            if point_in_polygon(x, y, points):
                label[i, j, 0] = 1
                label[i, j, 1:] = points
    return label
