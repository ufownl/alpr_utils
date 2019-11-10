import cv2
import random
import numpy as np
import mxnet as mx
from .chinese import random_plate

def gauss_blur(image, level):
    return cv2.blur(image, (level * 2 + 1, level * 2 + 1))

def gauss_noise(image):
    for i in range(image.shape[2]):
        c = image[:, :, i]
        diff = 255 - c.max();
        noise = np.random.normal(0, random.randint(1, 6), c.shape)
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        noise = diff * noise
        image[:, :, i] = c + noise.astype(np.uint8)
    return image

def fake_plate(smudge=None):
    draw = random_plate.Draw()
    plate, label = draw()
    if smudge:
        plate = smudge(plate)
    plate = gauss_blur(plate, random.randint(1, 8))
    plate = gauss_noise(plate)
    return mx.nd.array(plate), label


class Smudginess:
    def __init__(self, smu="fake/res/smu.png"):
        self._smu = cv2.imread(smu)

    def __call__(self, raw):
        y = random.randint(0, self._smu.shape[0] - raw.shape[0])
        x = random.randint(0, self._smu.shape[1] - raw.shape[1])
        texture = self._smu[y:y+raw.shape[0], x:x+raw.shape[1]]
        return cv2.bitwise_not(cv2.bitwise_and(cv2.bitwise_not(raw), texture))
