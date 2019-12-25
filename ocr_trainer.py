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
import time
import random
import argparse
import mxnet as mx
from dataset import ocr_batches
from utils import Vocabulary
from ocr_net import OcrNet


def train(max_epochs, epoch_size, learning_rate, batch_size, dims, max_hw, max_len, sgd, context):
    print("Loading dataset...", flush=True)
    vocab = Vocabulary()
    if os.path.isfile("model/vocabulary.json"):
        vocab.load("model/vocabulary.json")
    else:
        vocab.load("data/train/vocabulary.json")
        vocab.save("model/vocabulary.json")
    print("Vocabulary size: ", vocab.size())

    model = OcrNet(max_hw, vocab.size(), max_len)
    loss = mx.gluon.loss.SoftmaxCrossEntropyLoss(axis=2)
    if os.path.isfile("model/ocr_net.params"):
        model.load_parameters("model/ocr_net.params", ctx=context)
    else:
        model.initialize(mx.init.Xavier(), ctx=context)

    print("Learning rate: ", learning_rate)
    if sgd:
        print("Optimizer: SGD")
        trainer = mx.gluon.Trainer(model.collect_params(), "SGD", {
            "learning_rate": learning_rate,
            "momentum": 0.5
        })
    else:
        print("Optimizer: Nadam")
        trainer = mx.gluon.Trainer(model.collect_params(), "Nadam", {
            "learning_rate": learning_rate
        })
    if os.path.isfile("model/ocr_net.state"):
        trainer.load_states("model/ocr_net.state")

    print("Traning...", flush=True)
    for epoch in range(max_epochs):
        ts = time.time()

        training_total_L = 0.0
        training_batches = 0
        num_correct = 0
        num_inst = 0
        for x, tgt, tgt_len, lbl in ocr_batches(epoch_size, batch_size, dims, max_hw, vocab, max_len, context):
            training_batches += 1
            with mx.autograd.record():
                y, self_attn, context_attn = model(x, tgt, tgt_len)
                L = loss(y, lbl, mx.nd.not_equal(lbl, vocab.char2idx("<PAD>")).expand_dims(-1))
                L.backward()
            trainer.step(x.shape[0])
            training_batch_L = mx.nd.mean(L).asscalar()
            if training_batch_L != training_batch_L:
                raise ValueError()
            training_total_L += training_batch_L
            pred_lbl = mx.nd.argmax(y, axis=-1)
            batch_num_correct = (pred_lbl == lbl).sum().asscalar()
            batch_num_inst = len(pred_lbl.reshape((-1))) - (lbl == vocab.char2idx("<PAD>")).sum().asscalar()
            print("[Epoch %d  Batch %d]  batch_loss %.10f  average_loss %.10f  accuracy %.10f  elapsed %.2fs" % (
                epoch, training_batches, training_batch_L, training_total_L / training_batches, batch_num_correct / batch_num_inst, time.time() - ts
            ), flush=True)
            num_correct += batch_num_correct
            num_inst += batch_num_inst
        training_avg_L = training_total_L / training_batches

        print("[Epoch %d]  training_loss %.10f  validation_loss 0.0  accuracy %.10f  duration %.2fs" % (
            epoch + 1, training_avg_L, num_correct / num_inst, time.time() - ts
        ), flush=True)

        model.save_parameters("model/ocr_net.params")
        trainer.save_states("model/ocr_net.state")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a OCR network trainer.")
    parser.add_argument("--max_epochs", help="set the max epochs (default: 100)", type=int, default=100)
    parser.add_argument("--epoch_size", help="set the number of batches per epoch (default: 1000)", type=int, default=1000)
    parser.add_argument("--learning_rate", help="set the learning rate (default: 0.0001)", type=float, default=0.0001)
    parser.add_argument("--batch_size", help="set the batch size (default: 32)", type=int, default=32)
    parser.add_argument("--dims", help="set the sample dimentions (default: 208)", type=int, default=208)
    parser.add_argument("--img_w", help="set the max width of input images (default: 144)", type=int, default=144)
    parser.add_argument("--img_h", help="set the max height of input images (default: 48)", type=int, default=48)
    parser.add_argument("--seq_len", help="set the max length of output sequences (default: 8)", type=int, default=8)
    parser.add_argument("--sgd", help="using sgd optimizer", action="store_true")
    parser.add_argument("--device_id", help="select device that the model using (default: 0)", type=int, default=0)
    parser.add_argument("--gpu", help="using gpu acceleration", action="store_true")
    args = parser.parse_args()

    if args.gpu:
        context = mx.gpu(args.device_id)
    else:
        context = mx.cpu(args.device_id)

    train(args.max_epochs, args.epoch_size, args.learning_rate, args.batch_size, args.dims, (args.img_h, args.img_w), args.seq_len, args.sgd, context)
