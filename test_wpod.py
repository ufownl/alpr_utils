import argparse
import mxnet as mx
import matplotlib.pyplot as plt
from dataset import load_image, visualize, color_normalize
from utils import plate_reconstruct
from wpod_net import WpodNet


def test(images, dims, threshold, context):
    print("Loading model...")
    model = WpodNet()
    model.load_parameters("model/wpod_net.params", ctx=context)
    for path in images:
        img = load_image(path)
        h = img.shape[0]
        w = img.shape[1]
        f = min(288 * max(h, w) / min(h, w), 608) / min(h, w)
        img = mx.image.imresize(
            img,
            int(w * f) + (0 if w % 16 == 0 else 16 - w % 16),
            int(h * f) + (0 if h % 16 == 0 else 16 - h % 16)
        )
        x = color_normalize(img).transpose((2, 0, 1)).expand_dims(0)
        y = model(x.as_in_context(context))
        probs = y[0, :, :, 0]
        affines = y[0, :, :, 2:]
        labels, plates = plate_reconstruct(img, probs, affines, dims, 16, threshold)
        plt.subplot(2, 1, 1)
        visualize(img, [(pts.reshape((-1)).asnumpy().tolist(), str(prob)) for pts, prob in labels])
        plt.subplot(2, 1, 2)
        visualize(probs > threshold)
        plt.show()
        for i, plate in enumerate(plates):
            plt.subplot(len(plates), 1, i + 1)
            visualize(plate)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a cycle_gan tester.")
    parser.add_argument("images", metavar="IMG", help="path of the image file[s]", type=str, nargs="+")
    parser.add_argument("--dims", help="set the sample dimentions (default: 208)", type=int, default=208)
    parser.add_argument("--threshold", help="set the positive threshold (default: 0.9)", type=float, default=0.9)
    parser.add_argument("--device_id", help="select device that the model using (default: 0)", type=int, default=0)
    parser.add_argument("--gpu", help="using gpu acceleration", action="store_true")
    args = parser.parse_args()

    if args.gpu:
        context = mx.gpu(args.device_id)
    else:
        context = mx.cpu(args.device_id)

    test(args.images, args.dims, args.threshold, context)
