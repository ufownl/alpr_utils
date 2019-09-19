import os
import time
import random
import argparse
import mxnet as mx
from dataset import load_dataset, batches
from wpod_net import WpodNet, wpod_loss


def train(max_epochs, learning_rate, batch_size, dims, sgd, context):
    print("Loading dataset...", flush=True)
    dataset = load_dataset("data/train")
    split = int(len(dataset) * 0.9)
    training_set = dataset[:split]
    print("Training set: ", len(training_set))
    validating_set = dataset[split:]
    print("Validating set: ", len(validating_set))

    model = WpodNet()
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
        print("Optimizer: Adam")
        trainer = mx.gluon.Trainer(model.collect_params(), "Adam", {
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
        for x, label in batches(training_set, batch_size, dims, context):
            training_batches += 1
            with mx.autograd.record():
                y = model(x)
                L = wpod_loss(y, label)
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

        validating_total_L = 0.0
        validating_batches = 0
        for x, label in batches(validating_set, batch_size, dims, context):
            validating_batches += 1
            y = model(x)
            L = wpod_loss(y, label)
            validating_batch_L = mx.nd.mean(L).asscalar()
            if validating_batch_L != validating_batch_L:
                raise ValueError()
            validating_total_L += validating_batch_L
        validating_avg_L = validating_total_L / validating_batches

        print("[Epoch %d]  training_loss %.10f  validating_loss %.10f  duration %.2fs" % (
            epoch + 1, training_avg_L, validating_avg_L, time.time() - ts
        ), flush=True)

        model.save_parameters("model/wpod_net.params")
        trainer.save_states("model/wpod_net.state")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a transformer_coupletbot trainer.")
    parser.add_argument("--max_epochs", help="set the max epochs (default: 100)", type=int, default=100)
    parser.add_argument("--learning_rate", help="set the learning rate (default: 0.001)", type=float, default=0.001)
    parser.add_argument("--batch_size", help="set the batch size (default: 32)", type=int, default=32)
    parser.add_argument("--dims", help="set the sample dimentions (default: 208)", type=int, default=208)
    parser.add_argument("--sgd", help="using sgd optimizer", action="store_true")
    parser.add_argument("--device_id", help="select device that the model using (default: 0)", type=int, default=0)
    parser.add_argument("--gpu", help="using gpu acceleration", action="store_true")
    args = parser.parse_args()

    if args.gpu:
        context = mx.gpu(args.device_id)
    else:
        context = mx.cpu(args.device_id)

    train(args.max_epochs, args.learning_rate, args.batch_size, args.dims, args.sgd, context)
