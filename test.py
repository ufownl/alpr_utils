import math
import argparse
import mxnet as mx
import matplotlib.pyplot as plt
from gluoncv import model_zoo, data
from dataset import load_image, visualize, color_normalize
from utils import plate_labels, reconstruct_plates, Vocabulary
from wpod_net import WpodNet
from ocr_net import OcrNet


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


def recognize_plate(vocab, ocr, plate, beam_size, context):
    x = color_normalize(plate).transpose((2, 0, 1)).expand_dims(0)
    enc_y, enc_self_attn = ocr.encode(x.as_in_context(context))
    sequence = [vocab.char2idx("<GO>")]
    while True:
        target = mx.nd.array(sequence, ctx=context).reshape((1, -1))
        tgt_len = mx.nd.array([len(sequence)], ctx=context)
        output, dec_self_attn, context_attn = ocr.decode(target, tgt_len, enc_y)
        index = mx.nd.argmax(output, axis=2)
        char_token = index[0, -1].asscalar()
        sequence += [char_token]
        if char_token == vocab.char2idx("<EOS>"):
            break;
        print(vocab.idx2char(char_token), end="", flush=True)
    print("")
    print(sequence)


def detect_plate(wpod, vocab, ocr, raw, dims, threshold, plt_hw, context):
    h = raw.shape[0]
    w = raw.shape[1]
    f = min(288 * max(h, w) / min(h, w), 608) / min(h, w)
    img = mx.image.imresize(
        raw,
        int(w * f) + (0 if w % 16 == 0 else 16 - w % 16),
        int(h * f) + (0 if h % 16 == 0 else 16 - h % 16)
    )
    x = color_normalize(img).transpose((2, 0, 1)).expand_dims(0)
    y = wpod(x.as_in_context(context))
    probs = y[0, :, :, 0]
    affines = y[0, :, :, 2:]
    labels = plate_labels(img, probs, affines, dims, 16, threshold)
    plates = reconstruct_plates(raw, [pts for pts, _ in labels], (plt_hw[1], plt_hw[0]))
    plt.subplot(math.ceil((len(plates) + 2) / 2), 2, 1)
    visualize(img, [(pts.reshape((-1)).asnumpy().tolist(), str(prob)) for pts, prob in labels])
    plt.subplot(math.ceil((len(plates) + 2) / 2), 2, 2)
    visualize(probs > threshold)
    for i, plate in enumerate(plates):
        plt.subplot(math.ceil((len(plates) + 2) / 2), 2, i + 3)
        visualize(plate)
        print("plate[%d]:" % i)
        recognize_plate(vocab, ocr, plate, 5, context)
    plt.show()


def test(images, dims, threshold, plt_hw, seq_len, no_yolo, context):
    print("Loading model...")
    if not no_yolo:
        yolo = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True, ctx=context)
    wpod = WpodNet()
    wpod.load_parameters("model/wpod_net.params", ctx=context)
    vocab = Vocabulary()
    vocab.load("model/vocabulary.json")
    ocr = OcrNet(plt_hw, vocab.size(), seq_len)
    ocr.load_parameters("model/ocr_net.params", ctx=context)
    for path in images:
        print(path)
        if no_yolo:
            raw = load_image(path)
            detect_plate(wpod, vocab, ocr, raw, dims, threshold, plt_hw, context)
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
            for i, raw in enumerate(automobiles):
                print("automobile[%d]:" % i)
                detect_plate(wpod, vocab, ocr, raw, dims, threshold, plt_hw, context)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a ALPR tester.")
    parser.add_argument("images", metavar="IMG", help="path of the image file[s]", type=str, nargs="+")
    parser.add_argument("--dims", help="set the sample dimentions (default: 208)", type=int, default=208)
    parser.add_argument("--threshold", help="set the positive threshold (default: 0.9)", type=float, default=0.9)
    parser.add_argument("--plt_w", help="set the max width of output plate images (default: 384)", type=int, default=384)
    parser.add_argument("--plt_h", help="set the max height of output plate images (default: 128)", type=int, default=128)
    parser.add_argument("--seq_len", help="set the max length of output sequences (default: 8)", type=int, default=8)
    parser.add_argument("--device_id", help="select device that the model using (default: 0)", type=int, default=0)
    parser.add_argument("--no_yolo", help="do not extract automobiles using YOLOv3", action="store_true")
    parser.add_argument("--gpu", help="using gpu acceleration", action="store_true")
    args = parser.parse_args()

    if args.gpu:
        context = mx.gpu(args.device_id)
    else:
        context = mx.cpu(args.device_id)

    test(args.images, args.dims, args.threshold, (args.plt_h, args.plt_w), args.seq_len, args.no_yolo, context)
