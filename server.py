import io
import re
import png
import math
import json
import base64
import argparse
import http.server
import cgi
import mxnet as mx
from gluoncv import model_zoo, data
from dataset import color_normalize
from utils import plate_labels, reconstruct_plates
from wpod_net import WpodNet


class AlprHandler(http.server.BaseHTTPRequestHandler):
    _path_pattern = re.compile("^(/[^?\s]*)(\?\S*)?$")

    def do_POST(self):
        m = self._path_pattern.match(self.path)
        if not m or m.group(0) != self.path:
            self.send_error(http.HTTPStatus.BAD_REQUEST)
            return
        if m.group(1) == "/alpr_utils/wpod":
            form = cgi.FieldStorage(
                fp = self.rfile,
                headers = self.headers,
                environ = {
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers["Content-Type"]
                }
            )
            contents = [form[k].value for k in form.keys()]
            self.send_response(http.HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps([self._call_wpod(mx.image.imdecode(c)) for c in contents]).encode())
        else:
            self.send_error(http.HTTPStatus.NOT_FOUND)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST")
        self.send_header("Access-Control-Allow-Headers", "Keep-Alive,User-Agent,Authorization,Content-Type")
        super(AlprHandler, self).end_headers()

    def _call_wpod(self, raw):
        if self.yolo:
            raw = mx.image.resize_short(raw, 512)
            x = color_normalize(raw).transpose((2, 0, 1)).expand_dims(0)
            classes, scores, bboxes = self.yolo(x)
            automobiles = [
                fixed_crop(mx.nd.array(raw), bboxes[0, i])
                for i in range(classes.shape[1])
                    if (self.yolo.classes[int(classes[0, i].asscalar())] == 'car' or
                        self.yolo.classes[int(classes[0, i].asscalar())] == 'bus') and
                        scores[0, i].asscalar() > 0.5
            ]
            return [dict(automobile=png_encode(raw), plates=self._extract_plates(raw)) for raw in automobiles]
        else:
            return [dict(automobile=png_encode(raw), plates=self._extract_plates(raw))]

    def _extract_plates(self, raw):
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
        plates = reconstruct_plates(raw, labels)
        return [dict(plate=png_encode(plates[i]), score=labels[i][1].item()) for i in range(len(labels))]


def config_handler(context, dims, threshold, wpod, yolo=None):
    AlprHandler.context = context
    AlprHandler.dims = dims
    AlprHandler.threshold = threshold
    AlprHandler.wpod = wpod
    AlprHandler.yolo = yolo
    return AlprHandler


def png_encode(img):
    height = img.shape[0]
    width = img.shape[1]
    img = img.astype("uint8").reshape((-1, width * 3))
    f = io.BytesIO()
    w = png.Writer(width, height, greyscale=False)
    w.write(f, img.asnumpy())
    return base64.b64encode(f.getvalue()).decode()


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
    parser = argparse.ArgumentParser(description="Start a ALPR server.")
    parser.add_argument("--dims", help="set the sample dimentions (default: 208)", type=int, default=208)
    parser.add_argument("--threshold", help="set the positive threshold (default: 0.9)", type=float, default=0.9)
    parser.add_argument("--addr", help="set address of ALPR server (default: 0.0.0.0)", type=str, default="0.0.0.0")
    parser.add_argument("--port", help="set port of ALPR server (default: 80)", type=int, default=80)
    parser.add_argument("--device_id", help="select device that the model using (default: 0)", type=int, default=0)
    parser.add_argument("--no_yolo", help="Do not extract automobiles using YOLOv3", action="store_true")
    parser.add_argument("--gpu", help="using gpu acceleration", action="store_true")
    args = parser.parse_args()

    if args.gpu:
        context = mx.gpu(args.device_id)
    else:
        context = mx.cpu(args.device_id)

    print("Loading model...", flush=True)
    wpod = WpodNet()
    wpod.load_parameters("model/wpod_net.params", ctx=context)
    if args.no_yolo:
        handler = config_handler(context, args.dims, args.threshold, wpod)
    else:
        yolo = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True, ctx=context)
        handler = config_handler(context, args.dims, args.threshold, wpod, yolo)

    httpd = http.server.HTTPServer((args.addr, args.port), handler)
    httpd.serve_forever()


if __name__ == "__main__":
    _main()
