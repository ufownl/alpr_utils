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
    # scale the coordinates of points to [0, 1]
    points = points / dims
    # random flip
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        points[0] = 1 - points[0]
        points = points[..., [1, 0, 3, 2]]
    # color augment
    hsv_mod = (np.random.rand(3).astype(np.float32) - 0.5) * 0.3
    hsv_mod[0] *= 360
    image = hsv_transform(image, hsv_mod)
    return mx.nd.array(image), np.asarray(points).reshape((-1,)).tolist()

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

def reconstruct_plates(image, labels, out_size=(240, 80)):
    wh = np.array([[image.shape[1]], [image.shape[0]]])
    plates = []
    for pts, _ in labels:
        pts = points_matrix(pts.asnumpy() * wh)
        t_pts = rect_matrix(0, 0, out_size[0], out_size[1])
        m = transform_matrix(pts, t_pts)
        plate = cv2.warpPerspective(image.astype("uint8").asnumpy(), m, out_size)
        plates.append(mx.nd.array(plate))
    return plates
