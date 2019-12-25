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


import os
import json
import random
import mxnet as mx
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool
from utils import Vocabulary, apply_plate, augment_sample, color_normalize, object_label, reconstruct_plates
from fake.utils import Smudginess, fake_plate


def load_dataset(root, filename="dataset.json"):
    with open(os.path.join(root, filename)) as f:
        return [(os.path.join(root, data["image"]), data["points"], data["plate"]) for data in json.loads(f.read())]

def load_image(path):
    with open(path, "rb") as f:
        buf = f.read()
    return mx.image.imdecode(buf)

def wpod_batches(dataset, batch_size, dims, fake, ctx):
    batches = len(dataset) // batch_size
    if batches * batch_size < len(dataset):
        batches += 1
    sampler = WpodSampler(dims, fake)
    with Pool(cpu_count() * 2) as p:
        for i in range(batches):
            start = i * batch_size
            samples = p.map(sampler, dataset[start:start+batch_size])
            images, labels = zip(*samples)
            yield mx.nd.concat(*images, dim=0).as_in_context(ctx), mx.nd.concat(*labels, dim=0).as_in_context(ctx)

def ocr_batches(dataset, batch_size, dims, out_hw, vocab, max_len, ctx):
    if type(dataset) is int:
        batches = dataset
    else:
        batches = len(dataset) // batch_size
        if batches * batch_size < len(dataset):
            batches += 1
    sampler = OcrSampler(dims, out_hw, vocab)
    with Pool(cpu_count() * 2) as p:
        for i in range(batches):
            if type(dataset) is int:
                samples = p.map(sampler, [None] * batch_size)
            else:
                start = i * batch_size
                samples = p.map(sampler, dataset[start:start+batch_size])
            imgs, tgt_tok, tgt_len = zip(*samples)
            tgt_bat = mx.nd.array(pad_batch(add_sent_prefix(tgt_tok, vocab), vocab, max_len + 1), ctx=ctx)
            tgt_len_bat = mx.nd.array(tgt_len, ctx=ctx) + 1
            lbl_bat = mx.nd.array(pad_batch(add_sent_suffix(tgt_tok, vocab), vocab, max_len + 1), ctx=ctx)
            yield mx.nd.concat(*imgs, dim=0).as_in_context(ctx), tgt_bat, tgt_len_bat, lbl_bat

def add_sent_prefix(batch, vocab):
    return [[vocab.char2idx("<GO>")] + sent for sent in batch]

def add_sent_suffix(batch, vocab):
    return [sent + [vocab.char2idx("<EOS>")] for sent in batch]

def pad_batch(batch, vocab, seq_len):
    return [sent + [vocab.char2idx("<PAD>")] * (seq_len - len(sent)) for sent in batch]

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

def reconstruct_color(img):
    mean = mx.nd.array([0.485, 0.456, 0.406])
    std = mx.nd.array([0.229, 0.224, 0.225])
    return ((img * std + mean) * 255).astype("uint8")


class WpodSampler:
    def __init__(self, dims, fake):
        self._dims = dims
        self._fake = fake
        if fake > 0:
            self._smudge = Smudginess()

    def __call__(self, data):
        img = load_image(data[0])
        if random.random() < self._fake:
            fake, _ = fake_plate(self._smudge)
            img = apply_plate(img, data[1], fake)
        img, pts = augment_sample(img, data[1], self._dims)
        img = color_normalize(img)
        lbl = object_label(pts, self._dims, 16)
        return img.transpose((2, 0, 1)).expand_dims(0), lbl.expand_dims(0)


class OcrSampler:
    def __init__(self, dims, out_hw, vocab):
        self._smudge = Smudginess()
        self._dims = dims
        self._out_hw = out_hw
        self._vocab = vocab

    def __call__(self, data):
        if data:
            img = load_image(data[0])
            img, pts = augment_sample(img, data[1], self._dims, 0.0)
            img = reconstruct_plates(img, [mx.nd.array(pts).reshape((2, 4))], (self._out_hw[1], self._out_hw[0]))[0]
            pts = [val + random.uniform(-0.1, 0.1) for val in [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]]
            lbl = data[2]
        else:
            img, lbl = fake_plate(self._smudge)
            pts = [val + random.uniform(-0.1, 0.1) for val in [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]]
            img, pts = augment_sample(img, pts, self._dims, 0.0)
        plt = reconstruct_plates(img, [mx.nd.array(pts).reshape((2, 4))], (self._out_hw[1], self._out_hw[0]))[0]
        plt = color_normalize(plt)
        return plt.transpose((2, 0, 1)).expand_dims(0), [self._vocab.char2idx(ch) for ch in lbl], len(lbl)


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
    for batches, (images, labels) in enumerate(wpod_batches(dataset, 4, 208, 0.5, mx.cpu())):
        print("batch preview: ", images, labels)
        for i in range(images.shape[0]):
            plt.subplot(2, images.shape[0], i + 1)
            visualize(reconstruct_color(images.transpose((0, 2, 3, 1))[i]))
            plt.subplot(2, images.shape[0], i + images.shape[0] + 1)
            visualize(labels[i, :, :, 0] * 255)
        plt.show()
        if batches >= 4:
            break
    vocab = Vocabulary()
    vocab.load("data/train/vocabulary.json")
    print("vocab size: ", vocab.size())
    for batches, (imgs, tgt, tgt_len, lbl) in enumerate(ocr_batches(5, 4, 208, (48, 144), vocab, 8, mx.cpu())):
        print("batch preview: ", imgs, tgt, tgt_len, lbl)
        for i in range(imgs.shape[0]):
            plt.subplot(1, imgs.shape[0], i + 1)
            visualize(reconstruct_color(imgs.transpose((0, 2, 3, 1))[i]))
        plt.show()
    for batches, (imgs, tgt, tgt_len, lbl) in enumerate(ocr_batches(dataset, 4, 208, (48, 144), vocab, 8, mx.cpu())):
        print("batch preview: ", imgs, tgt, tgt_len, lbl)
        for i in range(imgs.shape[0]):
            plt.subplot(1, imgs.shape[0], i + 1)
            visualize(reconstruct_color(imgs.transpose((0, 2, 3, 1))[i]))
        plt.show()
        if batches >= 4:
            break
