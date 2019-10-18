import math
import argparse
import mxnet as mx
import matplotlib.pyplot as plt
from gluoncv import model_zoo, data
from dataset import load_image, visualize, color_normalize
from utils import plate_labels, reconstruct_plates
from wpod_net import WpodNet


def fixed_crop(raw, bbox):
    x0 = max(int(bbox[0].asscalar()), 0)
    x0 = min(int(x0), raw.shape[1])
    y0 = max(int(bbox[1].asscalar()), 0)
    y0 = min(int(y0), raw.shape[0])
    x1 = max(int(bbox[2].asscalar()), 0)
    x1 = min(int(x1), raw.shape[1])
    y1 = max(int(bbox[3].asscalar()), 0)
    y1 = min(int(y1), raw.shape[0])
    return mx.image.fixed_crop(raw, x0, y0, x1 - x0, y1 - y0)


def detect_plate(model, raw, dims, threshold, context):
    h = raw.shape[0]
    w = raw.shape[1]
    f = min(288 * max(h, w) / min(h, w), 608) / min(h, w)
    img = mx.image.imresize(
        raw,
        int(w * f) + (0 if w % 16 == 0 else 16 - w % 16),
        int(h * f) + (0 if h % 16 == 0 else 16 - h % 16)
    )
    x = color_normalize(img).transpose((2, 0, 1)).expand_dims(0)
    y = model(x.as_in_context(context))
    probs = y[0, :, :, 0]
    affines = y[0, :, :, 2:]
    labels = plate_labels(img, probs, affines, dims, 16, threshold)
    plates = reconstruct_plates(raw, labels)
    plt.subplot(math.ceil((len(plates) + 2) / 2), 2, 1)
    visualize(img, [(pts.reshape((-1)).asnumpy().tolist(), str(prob)) for pts, prob in labels])
    plt.subplot(math.ceil((len(plates) + 2) / 2), 2, 2)
    visualize(probs > threshold)
    for i, plate in enumerate(plates):
        plt.subplot(math.ceil((len(plates) + 2) / 2), 2, i + 3)
        visualize(plate)
    plt.show()


def test(images, dims, threshold, no_yolo, context):
    print("Loading model...")
    if not no_yolo:
        yolo = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True, ctx=context)
    model = WpodNet()
    model.load_parameters("model/wpod_net.params", ctx=context)
    for path in images:
        print(path)
        if no_yolo:
            raw = load_image(path)
            detect_plate(model, raw, dims, threshold, context)
        else:
            x, raw = data.transforms.presets.yolo.load_test(path, short=512)
            classes, scores, bboxes = yolo(x.as_in_context(context))
            automobiles = [
                fixed_crop(mx.nd.array(raw), bboxes[0, i])
                for i in range(classes.shape[1])
                    if (yolo.classes[int(classes[0, i].asscalar())] == 'car' or
                        yolo.classes[int(classes[0, i].asscalar())] == 'bus') and
                        scores[0, i].asscalar() > 0.5
            ]
            for raw in automobiles:
                detect_plate(model, raw, dims, threshold, context)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a WPOD tester.")
    parser.add_argument("images", metavar="IMG", help="path of the image file[s]", type=str, nargs="+")
    parser.add_argument("--dims", help="set the sample dimentions (default: 208)", type=int, default=208)
    parser.add_argument("--threshold", help="set the positive threshold (default: 0.9)", type=float, default=0.9)
    parser.add_argument("--device_id", help="select device that the model using (default: 0)", type=int, default=0)
    parser.add_argument("--no_yolo", help="Do not extract automobiles using YOLOv3", action="store_true")
    parser.add_argument("--gpu", help="using gpu acceleration", action="store_true")
    args = parser.parse_args()

    if args.gpu:
        context = mx.gpu(args.device_id)
    else:
        context = mx.cpu(args.device_id)

    test(args.images, args.dims, args.threshold, args.no_yolo, context)
