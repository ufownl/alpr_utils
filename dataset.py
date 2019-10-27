import os
import json
import random
import mxnet as mx
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool
from utils import Smudginess, fake_plate, apply_fake_plate, augment_sample, object_label


def load_dataset(path):
    with open(os.path.join(path, "dataset.json")) as f:
        return [(os.path.join(path, data["image"]), data["points"], data["plate"]) for data in json.loads(f.read())]

def load_image(path):
    with open(path, "rb") as f:
        buf = f.read()
    return mx.image.imdecode(buf)

def batches(dataset, batch_size, dims, fake, ctx):
    batches = len(dataset) // batch_size
    if batches * batch_size < len(dataset):
        batches += 1
    sampler = Sampler(dims, fake)
    with Pool(cpu_count() * 2) as p:
        for i in range(batches):
            start = i * batch_size
            samples = p.map(sampler, dataset[start: start + batch_size])
            images, labels = zip(*samples)
            yield mx.nd.concat(*images, dim=0).as_in_context(ctx), mx.nd.concat(*labels, dim=0).as_in_context(ctx)

def visualize(image, labels=None):
    plt.imshow(image.astype("uint8").asnumpy())
    if not labels is None:
        for points, tag in labels:
            x = [points[i] * image.shape[1] for i in range(0, len(points) // 2)] + [points[0] * image.shape[1],]
            y = [points[i] * image.shape[0] for i in range(len(points) // 2, len(points))] + [points[len(points) // 2] * image.shape[0],]
            plt.plot(x, y, "r")
            if not tag is None:
                plt.text(min(x), min(y) - 10, tag, bbox=dict(facecolor="green", alpha=0.5), fontdict=dict(color="white", size=8))
    plt.axis("off")

def color_normalize(img):
    return mx.image.color_normalize(
        img.astype("float32") / 255,
        mean = mx.nd.array([0.485, 0.456, 0.406]),
        std = mx.nd.array([0.229, 0.224, 0.225])
    )

def reconstruct_color(img):
    mean = mx.nd.array([0.485, 0.456, 0.406])
    std = mx.nd.array([0.229, 0.224, 0.225])
    return ((img * std + mean) * 255).astype("uint8")


class Sampler:
    def __init__(self, dims, fake):
        self._dims = dims
        self._fake = fake
        if fake > 0:
            self._smudge = Smudginess()

    def __call__(self, data):
        img = load_image(data[0])
        if random.random() < self._fake:
            fplt, flbl = fake_plate(self._smudge)
            img = apply_fake_plate(img, data[1], fplt)
        img, pts = augment_sample(img, data[1], self._dims)
        img = color_normalize(img)
        lbl = object_label(pts, self._dims, 16)
        return img.transpose((2, 0, 1)).expand_dims(0), lbl.expand_dims(0)


if __name__ == "__main__":
    dataset = load_dataset("data/train")
    print("dataset size: ", len(dataset))
    print("dataset preview: ", dataset[:10])
    path, points, plate = dataset[0]
    image = load_image(path)
    image, points = augment_sample(image, points, 208)
    label = object_label(points, 208, 16)
    plt.subplot(1, 2, 1)
    visualize(image, [(points, plate)])
    plt.subplot(1, 2, 2)
    visualize(label[:, :, 0])
    plt.show()
    for images, labels in batches(dataset, 4, 208, 0.5, mx.cpu()):
        print("batches preview: ", images, labels)
        for i in range(images.shape[0]):
            plt.subplot(2, images.shape[0], i + 1)
            visualize(reconstruct_color(images.transpose((0, 2, 3, 1))[i]))
            plt.subplot(2, images.shape[0], i + images.shape[0] + 1)
            visualize(labels[i, :, :, 0] * 255)
        plt.show()
