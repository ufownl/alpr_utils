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


import io
import re
import sys
import png
import math
import json
import base64
import argparse
import http.server
import cgi
import mxnet as mx
from gluoncv import model_zoo, data
from utils import color_normalize, plate_labels, reconstruct_plates, Vocabulary
from wpod_net import WpodNet
from ocr_net import OcrNet


class AlprHandler(http.server.BaseHTTPRequestHandler):
    _path_pattern = re.compile("^(/[^?\s]*)(\?\S*)?$")

    def do_POST(self):
        self._handle_request()
        sys.stdout.flush()
        sys.stderr.flush()

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST")
        self.send_header("Access-Control-Allow-Headers", "Keep-Alive,User-Agent,Authorization,Content-Type")
        super(AlprHandler, self).end_headers()

    def _handle_request(self):
        m = self._path_pattern.match(self.path)
        if not m or m.group(0) != self.path:
            self.send_error(http.HTTPStatus.BAD_REQUEST)
            return
        if m.group(1) == "/alpr_utils/run":
            form = cgi.FieldStorage(
                fp = self.rfile,
                headers = self.headers,
                environ = {
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers["Content-Type"]
                }
            )
            if not "image" in form:
                self.send_error(http.HTTPStatus.BAD_REQUEST)
                return
            yolo = False
            if "yolo" in form and form["yolo"].value == "true":
                yolo = True
            ret_vehicle = False
            if "vehicle" in form and form["vehicle"].value == "true":
                ret_vehicle = True
            ret_plate = False
            if "plate" in form and form["plate"].value == "true":
                ret_plate = True
            result = [
                dict(
                    image = png_encode(vehicle) if ret_vehicle else None,
                    plates = [
                        dict(
                            image = png_encode(image) if ret_plate else None,
                            text = text,
                            confidence = confidence
                        ) for image, text, confidence in plates
                    ]
                ) for vehicle, plates in self._alpr(mx.image.imdecode(form["image"].value).as_in_context(self.context), yolo)
            ]
            self.send_response(http.HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        else:
            self.send_error(http.HTTPStatus.NOT_FOUND)

    def _alpr(self, raw, yolo):
        if yolo:
            vehicles = self._detect_vehicles(raw)
        else:
            vehicles = [raw]
        return [
            (
                raw, sorted([
                    (img, txt, det_prob * rec_prob) for img, det_prob, (txt, rec_prob) in [
                        (img, prob, self._recognize_plate(img)) for img, prob in self._detect_plates(raw)
                    ]
                ], key=lambda tup: tup[2], reverse=True)
            ) for raw in vehicles
        ]

    def _detect_vehicles(self, raw):
        x, _ = data.transforms.presets.yolo.transform_test(raw, short=512)
        classes, scores, bboxes = self.yolo(x)
        bboxes[0, :, 0::2] = bboxes[0, :, 0::2] / x.shape[3] * raw.shape[1]
        bboxes[0, :, 1::2] = bboxes[0, :, 1::2] / x.shape[2] * raw.shape[0]
        return [
            fixed_crop(raw, bboxes[0, i]) for i in range(classes.shape[1])
                if (self.yolo.classes[int(classes[0, i].asscalar())] == 'car' or
                    self.yolo.classes[int(classes[0, i].asscalar())] == 'bus') and
                    scores[0, i].asscalar() > 0.5
        ]

    def _detect_plates(self, raw):
        h = raw.shape[0]
        w = raw.shape[1]
        f = min(288 * max(h, w) / min(h, w), 608) / min(h, w)
        img = mx.image.imresize(
            raw,
            int(w * f) + (0 if w % 16 == 0 else 16 - w % 16),
            int(h * f) + (0 if h % 16 == 0 else 16 - h % 16)
        )
        x = color_normalize(img).transpose((2, 0, 1)).expand_dims(0)
        y = self.wpod(x.as_in_context(self.context))
        probs = y[0, :, :, 0]
        affines = y[0, :, :, 2:]
        labels = plate_labels(img, probs, affines, self.dims, 16, self.threshold)
        plates = reconstruct_plates(raw, [pts for pts, _ in labels], (self.plt_hw[1], self.plt_hw[0]))
        return [(plates[i], labels[i][1].item()) for i in range(len(labels))]

    def _recognize_plate(self, img):
        x = color_normalize(img).transpose((2, 0, 1)).expand_dims(0)
        enc_y, self_attn = self.ocr.encode(x.as_in_context(self.context))
        sequences = [([self.vocab.char2idx("<GO>")], 0.0)]
        while True:
            candidates = []
            for seq, score in sequences:
                if seq[-1] == self.vocab.char2idx("<EOS>") or len(seq) >= self.seq_len + 2:
                    candidates.append((seq, score))
                else:
                    tgt = mx.nd.array(seq, ctx=self.context).reshape((1, -1))
                    tgt_len = mx.nd.array([len(seq)], ctx=self.context)
                    y, context_attn = self.ocr.decode(tgt, tgt_len, enc_y)
                    probs = mx.nd.softmax(y, axis=2)
                    beam = probs[0, -1].topk(k=self.beam_size, ret_typ="both")
                    for i in range(self.beam_size):
                        candidates.append((seq + [int(beam[1][i].asscalar())], score + math.log(beam[0][i].asscalar())))
            if len(candidates) <= len(sequences):
                break;
            sequences = sorted(candidates, key=lambda tup: tup[1], reverse=True)[:self.beam_size]
        scores = mx.nd.array([score for _, score in sequences], ctx=self.context)
        probs = mx.nd.softmax(scores)
        return "".join([self.vocab.idx2char(token) for token in sequences[0][0][1:-1]]), probs[0].asscalar()


def config_handler(context, dims, threshold, plt_hw, seq_len, beam_size, wpod, vocab, ocr, yolo=None):
    AlprHandler.context = context
    AlprHandler.dims = dims
    AlprHandler.threshold = threshold
    AlprHandler.plt_hw = plt_hw
    AlprHandler.seq_len = seq_len
    AlprHandler.beam_size = beam_size
    AlprHandler.wpod = wpod
    AlprHandler.vocab = vocab
    AlprHandler.ocr = ocr
    AlprHandler.yolo = yolo
    return AlprHandler


def png_encode(img):
    height = img.shape[0]
    width = img.shape[1]
    img = img.astype("uint8").reshape((-1, width * 3))
    f = io.BytesIO()
    w = png.Writer(width, height, greyscale=False)
    w.write(f, img.asnumpy())
    return "data:image/png;base64, " + base64.b64encode(f.getvalue()).decode()


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


def _main():
    parser = argparse.ArgumentParser(description="Start a ALPR demo server.")
    parser.add_argument("--dims", help="set the sample dimentions (default: 208)", type=int, default=208)
    parser.add_argument("--threshold", help="set the positive threshold (default: 0.9)", type=float, default=0.9)
    parser.add_argument("--plt_w", help="set the max width of output plate images (default: 144)", type=int, default=144)
    parser.add_argument("--plt_h", help="set the max height of output plate images (default: 48)", type=int, default=48)
    parser.add_argument("--seq_len", help="set the max length of output sequences (default: 8)", type=int, default=8)
    parser.add_argument("--beam_size", help="set the size of beam (default: 5)", type=int, default=5)
    parser.add_argument("--addr", help="set address of ALPR server (default: 0.0.0.0)", type=str, default="0.0.0.0")
    parser.add_argument("--port", help="set port of ALPR server (default: 80)", type=int, default=80)
    parser.add_argument("--device_id", help="select device that the model using (default: 0)", type=int, default=0)
    parser.add_argument("--gpu", help="using gpu acceleration", action="store_true")
    args = parser.parse_args()

    if args.gpu:
        context = mx.gpu(args.device_id)
    else:
        context = mx.cpu(args.device_id)

    print("This is ALPR demo server", flush=True)
    wpod = WpodNet()
    wpod.load_parameters("model/wpod_net.params", ctx=context)
    vocab = Vocabulary()
    vocab.load("model/vocabulary.json")
    ocr = OcrNet((args.plt_h, args.plt_w), vocab.size(), args.seq_len)
    ocr.load_parameters("model/ocr_net.params", ctx=context)
    yolo = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True, ctx=context)
    handler = config_handler(
        context = context,
        dims = args.dims,
        threshold = args.threshold,
        plt_hw = (args.plt_h, args.plt_w),
        seq_len = args.seq_len,
        beam_size = args.beam_size,
        wpod = wpod,
        vocab = vocab,
        ocr = ocr,
        yolo = yolo
    )

    httpd = http.server.HTTPServer((args.addr, args.port), handler)
    httpd.serve_forever()


if __name__ == "__main__":
    _main()
