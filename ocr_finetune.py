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
from dataset import load_dataset, ocr_batches
from utils import Vocabulary
from ocr_net import OcrNet


def finetune(max_epochs, learning_rate, batch_size, dims, max_hw, max_len, sgd, context):
    print("Loading dataset...", flush=True)
    dataset = load_dataset("data/train", "finetune.json")
    split = int(len(dataset) * 0.9)
    training_set = dataset[:split]
    print("Training set: ", len(training_set))
    validation_set = dataset[split:]
    print("Validation set: ", len(validation_set))

    print("Loading model...", flush=True)
    vocab = Vocabulary()
    vocab.load("model/vocabulary.json")
    print("Vocabulary size: ", vocab.size())
    model = OcrNet(max_hw, vocab.size(), max_len)
    loss = mx.gluon.loss.SoftmaxCrossEntropyLoss(axis=2)
    model.load_parameters("model/ocr_net.params", ctx=context)

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

        random.shuffle(training_set)
        training_total_L = 0.0
        training_batches = 0
        training_num_correct = 0
        training_num_inst = 0
        for x, tgt, tgt_len, lbl in ocr_batches(training_set, batch_size, dims, max_hw, vocab, max_len, context):
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
            training_num_correct += batch_num_correct
            training_num_inst += batch_num_inst
        training_avg_L = training_total_L / training_batches

        validation_total_L = 0.0
        validation_batches = 0
        validation_num_correct = 0
        validation_num_inst = 0
        for x, tgt, tgt_len, lbl in ocr_batches(validation_set, batch_size, dims, max_hw, vocab, max_len, context):
            validation_batches += 1
            y, self_attn, context_attn = model(x, tgt, tgt_len)
            L = loss(y, lbl, mx.nd.not_equal(lbl, vocab.char2idx("<PAD>")).expand_dims(-1))
            validation_batch_L = mx.nd.mean(L).asscalar()
            if validation_batch_L != validation_batch_L:
                raise ValueError()
            validation_total_L += validation_batch_L
            pred_lbl = mx.nd.argmax(y, axis=-1)
            batch_num_correct = (pred_lbl == lbl).sum().asscalar()
            validation_num_correct += (pred_lbl == lbl).sum().asscalar()
            validation_num_inst += len(pred_lbl.reshape((-1))) - (lbl == vocab.char2idx("<PAD>")).sum().asscalar()
        validation_avg_L = validation_total_L / validation_batches

        print("[Epoch %d]  training_loss %.10f  validation_loss %.10f  training_accuracy %.10f  validation_accuracy %.10f  duration %.2fs" % (
            epoch + 1, training_avg_L, validation_avg_L, training_num_correct / training_num_inst, validation_num_correct / validation_num_inst, time.time() - ts
        ), flush=True)

        model.save_parameters("model/ocr_net.params")
        trainer.save_states("model/ocr_net.state")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune the OCR network.")
    parser.add_argument("--max_epochs", help="set the max epochs (default: 10)", type=int, default=10)
    parser.add_argument("--learning_rate", help="set the learning rate (default: 1e-6)", type=float, default=1e-6)
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

    finetune(args.max_epochs, args.learning_rate, args.batch_size, args.dims, (args.img_h, args.img_w), args.seq_len, args.sgd, context)
