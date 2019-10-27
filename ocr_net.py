import math
import mxnet as mx
from transformer_utils import PositionalEncoding, TimingEncoding, EncoderLayer, AdaptiveComputationTime, Decoder


class ImageEmbedding(mx.gluon.nn.Block):
    def __init__(self, **kwargs):
        super(ImageEmbedding, self).__init__(**kwargs)
        self._features = mx.gluon.model_zoo.vision.resnet18_v2().features[:10]
        self._features.hybridize()

    def forward(self, x):
        y = self._features(x)
        return y.reshape((y.shape[0], y.shape[1], -1)).transpose((0, 2, 1))


class ImageEncoder(mx.gluon.nn.Block):
    def __init__(self, max_hw, layers, dims, heads, ffn_dims, dropout=0.0, **kwargs):
        super(ImageEncoder, self).__init__(**kwargs)
        h = math.ceil(max_hw[0] / 32)
        w = math.ceil(max_hw[1] / 32)
        with self.name_scope():
            self._embedding = ImageEmbedding()
            self._pos_encoding = PositionalEncoding(dims, h * w)
            self._time_encoding = TimingEncoding(dims, layers)
            self._encoder = EncoderLayer(dims, heads, ffn_dims, dropout)
            self._act = AdaptiveComputationTime(layers)

    def forward(self, x):
        y = self._embedding(x)
        seq_len = mx.nd.array([y.shape[1]] * y.shape[0], ctx=y.context)
        return self._act(self._encoder, self._pos_encoding, self._time_encoding, y, seq_len, None)


class OcrNet(mx.gluon.nn.Block):
    def __init__(self, max_hw, vocab_size, max_len, **kwargs):
        super(OcrNet, self).__init__(**kwargs)
        with self.name_scope():
            self._encoder = ImageEncoder(max_hw, 6, 512, 8, 2048, 0.5)
            self._decoder = Decoder(vocab_size, max_len + 1, 6, 512, 8, 2048, 0.5)
            self._output = mx.gluon.nn.Dense(vocab_size, flatten=False)

    def forward(self, img, tgt_seq, tgt_len):
        out, enc_self_attn = self.encode(img) 
        out, dec_self_attn, context_attn = self.decode(tgt_seq, tgt_len, out)
        return out, enc_self_attn, dec_self_attn, context_attn

    def encode(self, img):
        return self._encoder(img)

    def decode(self, seq, seq_len, enc_out):
        out, self_attn, context_attn = self._decoder(seq, seq_len, enc_out, None)
        out = self._output(out)
        return out, self_attn, context_attn


if __name__ == "__main__":
    x = mx.nd.zeros((4, 3, 128, 384))
    net = OcrNet((128, 384), 69, 8)
    net.initialize(mx.init.Xavier())
    print(net(mx.nd.zeros((4, 3, 128, 384)), mx.nd.zeros((4, 9)), mx.nd.ones((4,)) * 9))
