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
from dataset import load_dataset, wpod_batches
from wpod_net import WpodNet, WpodLoss


def train(max_epochs, learning_rate, batch_size, dims, fake, sgd, context):
    print("Loading dataset...", flush=True)
    dataset = load_dataset("data/train")
    split = int(len(dataset) * 0.9)
    training_set = dataset[:split]
    print("Training set: ", len(training_set))
    validation_set = dataset[split:]
    print("Validation set: ", len(validation_set))

    model = WpodNet()
    loss = WpodLoss()
    if os.path.isfile("model/wpod_net.params"):
        model.load_parameters("model/wpod_net.params", ctx=context)
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
    if os.path.isfile("model/wpod_net.state"):
        trainer.load_states("model/wpod_net.state")

    print("Traning...", flush=True)
    for epoch in range(max_epochs):
        ts = time.time()

        random.shuffle(training_set)
        training_total_L = 0.0
        training_batches = 0
        for x, label in wpod_batches(training_set, batch_size, dims, fake, context):
            training_batches += 1
            with mx.autograd.record():
                y = model(x)
                L = loss(y, label)
                L.backward()
            trainer.step(x.shape[0])
            training_batch_L = mx.nd.mean(L).asscalar()
            if training_batch_L != training_batch_L:
                raise ValueError()
            training_total_L += training_batch_L
            print("[Epoch %d  Batch %d]  batch_loss %.10f  average_loss %.10f  elapsed %.2fs" % (
                epoch, training_batches, training_batch_L, training_total_L / training_batches, time.time() - ts
            ), flush=True)
        training_avg_L = training_total_L / training_batches

        validation_total_L = 0.0
        validation_batches = 0
        for x, label in wpod_batches(validation_set, batch_size, dims, fake, context):
            validation_batches += 1
            y = model(x)
            L = loss(y, label)
            validation_batch_L = mx.nd.mean(L).asscalar()
            if validation_batch_L != validation_batch_L:
                raise ValueError()
            validation_total_L += validation_batch_L
        validation_avg_L = validation_total_L / validation_batches

        print("[Epoch %d]  training_loss %.10f  validation_loss %.10f  duration %.2fs" % (
            epoch + 1, training_avg_L, validation_avg_L, time.time() - ts
        ), flush=True)

        model.save_parameters("model/wpod_net.params")
        trainer.save_states("model/wpod_net.state")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a WPOD network trainer.")
    parser.add_argument("--max_epochs", help="set the max epochs (default: 100)", type=int, default=100)
    parser.add_argument("--learning_rate", help="set the learning rate (default: 0.0001)", type=float, default=0.0001)
    parser.add_argument("--batch_size", help="set the batch size (default: 32)", type=int, default=32)
    parser.add_argument("--dims", help="set the sample dimentions (default: 208)", type=int, default=208)
    parser.add_argument("--fake", help="set the probability of using a fake plate (default: 0.0)", type=float, default=0.0)
    parser.add_argument("--sgd", help="using sgd optimizer", action="store_true")
    parser.add_argument("--device_id", help="select device that the model using (default: 0)", type=int, default=0)
    parser.add_argument("--gpu", help="using gpu acceleration", action="store_true")
    args = parser.parse_args()

    if args.gpu:
        context = mx.gpu(args.device_id)
    else:
        context = mx.cpu(args.device_id)

    train(args.max_epochs, args.learning_rate, args.batch_size, args.dims, args.fake, args.sgd, context)
