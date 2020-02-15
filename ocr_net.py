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


import math
import backbone
import mxnet as mx
from transformer_utils import MultiHeadAttention, PositionalEncoding, TimingEncoding, PositionalWiseFeedForward, EncoderLayer, AdaptiveComputationTime


class ImageEmbedding(mx.gluon.nn.Block):
    def __init__(self, dims, **kwargs):
        super(ImageEmbedding, self).__init__(**kwargs)
        with self.name_scope():
            self._block = mx.gluon.nn.Sequential()
            backbone.add_layers(self._block, dims)

    def forward(self, x):
        y = self._block(x).transpose((0, 1, 3, 2))
        return y.reshape((y.shape[0], y.shape[1], -1)).transpose((0, 2, 1))


class ImageEncoder(mx.gluon.nn.Block):
    def __init__(self, max_hw, layers, dims, heads, ffn_dims, dropout=0.0, **kwargs):
        super(ImageEncoder, self).__init__(**kwargs)
        h = math.ceil(max_hw[0] / 16)
        w = math.ceil(max_hw[1] / 16)
        with self.name_scope():
            self._embedding = ImageEmbedding(dims)
            self._pos_encoding = PositionalEncoding(dims, h * w)
            self._time_encoding = TimingEncoding(dims, layers)
            self._encoder = EncoderLayer(dims, heads, ffn_dims, dropout)
            self._act = AdaptiveComputationTime(layers)

    def forward(self, x):
        y = self._embedding(x)
        seq_len = mx.nd.array([y.shape[1]] * y.shape[0], ctx=y.context)
        return self._act(self._encoder, self._pos_encoding, self._time_encoding, y, seq_len, None)


class LpDecoderLayer(mx.gluon.nn.Block):
    def __init__(self, dims, heads, ffn_dims, dropout=0.0, **kwargs):
        super(LpDecoderLayer, self).__init__(**kwargs)
        with self.name_scope():
            self._layer_norm = mx.gluon.nn.LayerNorm()
            self._context_attn = MultiHeadAttention(dims, heads, dropout)
            self._ffn = PositionalWiseFeedForward(dims, ffn_dims, dropout)

    def forward(self, x, enc_y, self_attn_mask, context_attn_mask):
        y, context_attn = self._context_attn(self._layer_norm(x), enc_y, enc_y, x, context_attn_mask)
        return self._ffn(self._layer_norm(y), y), None, context_attn


class LpDecoder(mx.gluon.nn.Block):
    def __init__(self, vocab_size, max_len, layers, dims, heads, ffn_dims, dropout=0.0, **kwargs):
        super(LpDecoder, self).__init__(**kwargs)
        with self.name_scope():
            self._embedding = mx.gluon.nn.Embedding(vocab_size, dims, weight_initializer=mx.init.Uniform(0.1))
            self._pos_encoding = PositionalEncoding(dims, max_len)
            self._time_encoding = TimingEncoding(dims, layers)
            self._decoder = LpDecoderLayer(dims, heads, ffn_dims, dropout)
            self._act = AdaptiveComputationTime(layers)

    def forward(self, x, seq_len, enc_y):
        y = self._embedding(x)
        return self._act(self._decoder, self._pos_encoding, self._time_encoding, y, seq_len, None, enc_y)


class OcrNet(mx.gluon.nn.Block):
    def __init__(self, max_hw, vocab_size, max_len, **kwargs):
        super(OcrNet, self).__init__(**kwargs)
        with self.name_scope():
            self._encoder = ImageEncoder(max_hw, 6, 256, 8, 2048, 0.5)
            self._decoder = LpDecoder(vocab_size, max_len + 1, 6, 256, 8, 2048, 0.5)
            self._output = mx.gluon.nn.Dense(vocab_size, flatten=False)

    def forward(self, img, tgt_seq, tgt_len):
        out, self_attn = self.encode(img)
        out, context_attn = self.decode(tgt_seq, tgt_len, out)
        return out, self_attn, context_attn

    def encode(self, img):
        return self._encoder(img)

    def decode(self, seq, seq_len, enc_out):
        out, self_attn, context_attn = self._decoder(seq, seq_len, enc_out)
        out = self._output(out)
        return out, context_attn


if __name__ == "__main__":
    x = mx.nd.zeros((4, 3, 48, 144))
    net = OcrNet((48, 144), 72, 8)
    net.initialize(mx.init.Xavier())
    print(net)
    print(net(mx.nd.zeros((4, 3, 48, 144)), mx.nd.zeros((4, 9)), mx.nd.ones((4,)) * 9))
