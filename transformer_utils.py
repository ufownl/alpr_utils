import sys
import numpy as np
import mxnet as mx


def padding_mask(seq_q, seq_k):
    mask = mx.nd.equal(seq_k, 0)
    mask = mask.expand_dims(1).broadcast_axes(1, seq_q.shape[1])
    return mask


def sequence_mask(seq):
    mask = mx.nd.array(np.triu(np.ones((seq.shape[1], seq.shape[1])), 1), ctx=seq.context)
    mask = mask.expand_dims(0).broadcast_axes(0, seq.shape[0])
    return mask


def mask_fill(a, mask, value):
    return a * mx.nd.logical_not(mask) + mask * value


class ScaledDotProductAttention(mx.gluon.nn.Block):
    def __init__(self, dropout=0.0, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        with self.name_scope():
            self._dropout = mx.gluon.nn.Dropout(dropout)

    def forward(self, q, k, v, scale, mask):
        attn = mx.nd.batch_dot(q, k, transpose_b=True)
        if not scale is None:
            attn = attn * scale
        if not mask is None:
            attn = mask_fill(attn, mask, -sys.maxsize-1)
        attn = mx.nd.softmax(attn, axis=2)
        attn = self._dropout(attn)
        return mx.nd.batch_dot(attn, v), attn


class MultiHeadAttention(mx.gluon.nn.Block):
    def __init__(self, dims, heads, dropout=0.0, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self._dims_per_head = dims // heads
        self._heads = heads
        with self.name_scope():
            self._dense_q = mx.gluon.nn.Dense(self._dims_per_head * heads, flatten=False)
            self._dense_k = mx.gluon.nn.Dense(self._dims_per_head * heads, flatten=False)
            self._dense_v = mx.gluon.nn.Dense(self._dims_per_head * heads, flatten=False)
            self._attention = ScaledDotProductAttention(dropout)
            self._dense_final = mx.gluon.nn.Dense(dims, flatten=False)
            self._dropout = mx.gluon.nn.Dropout(dropout)

    def forward(self, q, k, v, residual, mask):
        batch_size = q.shape[0]
        q = self._dense_q(q)
        k = self._dense_k(k)
        v = self._dense_v(v)
        q = q.reshape((batch_size, -1, self._heads, self._dims_per_head))
        q = q.transpose((0, 2, 1, 3))
        q = q.reshape((batch_size * self._heads, -1, self._dims_per_head))
        k = k.reshape((batch_size, -1, self._heads, self._dims_per_head))
        k = k.transpose((0, 2, 1, 3))
        k = k.reshape((batch_size * self._heads, -1, self._dims_per_head))
        v = v.reshape((batch_size, -1, self._heads, self._dims_per_head))
        v = v.transpose((0, 2, 1, 3))
        v = v.reshape((batch_size * self._heads, -1, self._dims_per_head))
        scale = self._dims_per_head ** -0.5
        if not mask is None:
            mask = mask.repeat(self._heads, axis=0)
        y, attn = self._attention(q, k, v, scale, mask)
        y = y.reshape((batch_size, self._heads, -1, self._dims_per_head))
        y = y.transpose((0, 2, 1, 3))
        y = y.reshape((batch_size, -1, self._dims_per_head * self._heads))
        y = self._dense_final(y)
        y = self._dropout(y)
        return y + residual, attn


class PositionalEncoding(mx.gluon.nn.Block):
    def __init__(self, dims, max_len, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self._dims = dims
        self._max_len = max_len + 1
        self._weight = None

    def forward(self, x, seq_len):
        if self._weight is None:
            self._weight = mx.nd.array([[pos / (10000 ** (2 * (i // 2) / self._dims)) for i in range(self._dims)] for pos in range(self._max_len)], ctx=x.context)
            self._weight[:, 0::2] = mx.nd.sin(self._weight[:, 0::2])
            self._weight[:, 1::2] = mx.nd.cos(self._weight[:, 1::2])
        seq_pos = mx.nd.array([list(range(1, int(l.asscalar()) + 1)) + [0] * (x.shape[1] - int(l.asscalar())) for l in seq_len], ctx=x.context)
        return mx.nd.Embedding(seq_pos, self._weight, self._max_len, self._dims)


class TimingEncoding(mx.gluon.nn.Block):
    def __init__(self, dims, max_len, **kwargs):
        super(TimingEncoding, self).__init__(**kwargs)
        self._dims = dims
        self._max_len = max_len
        self._weight = None

    def forward(self, x, t):
        if self._weight is None:
            self._weight = mx.nd.array([[pos / (10000 ** (2 * (i // 2) / self._dims)) for i in range(self._dims)] for pos in range(self._max_len)], ctx=x.context)
            self._weight[:, 0::2] = mx.nd.sin(self._weight[:, 0::2])
            self._weight[:, 1::2] = mx.nd.cos(self._weight[:, 1::2])
        seq_t = mx.nd.ones(x.shape[:2], ctx=x.context) * t
        return mx.nd.Embedding(seq_t, self._weight, self._max_len, self._dims)


class PositionalWiseFeedForward(mx.gluon.nn.Block):
    def __init__(self, dims, ffn_dims, dropout=0.0, **kwargs):
        super(PositionalWiseFeedForward, self).__init__(**kwargs)
        with self.name_scope():
            self._w1 = mx.gluon.nn.Conv1D(ffn_dims, 1)
            self._w2 = mx.gluon.nn.Conv1D(dims, 1)
            self._dropout = mx.gluon.nn.Dropout(dropout)

    def forward(self, x, residual):
        y = self._w2(mx.nd.relu(self._w1(x.transpose((0, 2, 1)))))
        y = self._dropout(y.transpose((0, 2, 1)))
        return y + residual


class EncoderLayer(mx.gluon.nn.Block):
    def __init__(self, dims, heads, ffn_dims, dropout=0.0, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        with self.name_scope():
            self._layer_norm = mx.gluon.nn.LayerNorm()
            self._self_attn = MultiHeadAttention(dims, heads, dropout)
            self._ffn = PositionalWiseFeedForward(dims, ffn_dims, dropout)

    def forward(self, x, mask):
        norm_x = self._layer_norm(x)
        y, attn = self._self_attn(norm_x, norm_x, norm_x, x, mask)
        return self._ffn(self._layer_norm(y), y), attn


class Encoder(mx.gluon.nn.Block):
    def __init__(self, vocab_size, max_len, layers, dims, heads, ffn_dims, dropout=0.0, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        with self.name_scope():
            self._embedding = mx.gluon.nn.Embedding(vocab_size, dims, weight_initializer=mx.init.Uniform(0.1))
            self._pos_encoding = PositionalEncoding(dims, max_len)
            self._time_encoding = TimingEncoding(dims, layers)
            self._encoder = EncoderLayer(dims, heads, ffn_dims, dropout)
            self._act = AdaptiveComputationTime(layers)

    def forward(self, x, seq_len):
        y = self._embedding(x)
        mask = padding_mask(x, x)
        return self._act(self._encoder, self._pos_encoding, self._time_encoding, y, seq_len, mask)


class DecoderLayer(mx.gluon.nn.Block):
    def __init__(self, dims, heads, ffn_dims, dropout=0.0, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        with self.name_scope():
            self._layer_norm = mx.gluon.nn.LayerNorm()
            self._self_attn = MultiHeadAttention(dims, heads, dropout)
            self._context_attn = MultiHeadAttention(dims, heads, dropout)
            self._ffn = PositionalWiseFeedForward(dims, ffn_dims, dropout)

    def forward(self, x, enc_y, self_attn_mask, context_attn_mask):
        norm_x = self._layer_norm(x)
        y, self_attn = self._self_attn(norm_x, norm_x, norm_x, x, self_attn_mask)
        y, context_attn = self._context_attn(self._layer_norm(y), enc_y, enc_y, y, context_attn_mask)
        return self._ffn(self._layer_norm(y), y), self_attn, context_attn


class Decoder(mx.gluon.nn.Block):
    def __init__(self, vocab_size, max_len, layers, dims, heads, ffn_dims, dropout=0.0, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        with self.name_scope():
            self._embedding = mx.gluon.nn.Embedding(vocab_size, dims, weight_initializer=mx.init.Uniform(0.1))
            self._pos_encoding = PositionalEncoding(dims, max_len)
            self._time_encoding = TimingEncoding(dims, layers)
            self._decoder = DecoderLayer(dims, heads, ffn_dims, dropout)
            self._act = AdaptiveComputationTime(layers)

    def forward(self, x, seq_len, enc_y, context_attn_mask):
        y = self._embedding(x)
        self_attn_mask = mx.nd.logical_or(padding_mask(x, x), sequence_mask(x))
        return self._act(self._decoder, self._pos_encoding, self._time_encoding, y, seq_len, self_attn_mask, enc_y, context_attn_mask)


class AdaptiveComputationTime(mx.gluon.nn.Block):
    def __init__(self, layers, threshold=0.9, **kwargs):
        super(AdaptiveComputationTime, self).__init__(**kwargs)
        self._layers = layers
        self._threshold = threshold
        with self.name_scope():
            self._p = mx.gluon.nn.Dense(1, activation="sigmoid", bias_initializer="ones", flatten=False)
            self._layer_norm = mx.gluon.nn.LayerNorm()

    def forward(self, fn, pos_encoding, time_encoding, x, seq_len, self_attn_mask, enc_y=None, context_attn_mask=None):
        halting_prob = mx.nd.zeros(x.shape[:2], ctx=x.context)
        remainders = mx.nd.zeros(x.shape[:2], ctx=x.context)
        updates = mx.nd.zeros(x.shape[:2], ctx=x.context)
        prev_state = mx.nd.zeros_like(x, ctx=x.context)
        y = x
        self_attns = []
        if not enc_y is None:
            context_attns = []
        t = 0
        while mx.nd.logical_and(halting_prob < self._threshold, updates < self._layers).sum() > 0:
            state = y + pos_encoding(y, seq_len) + time_encoding(y, t)
            p = self._p(state).flatten()
            running = halting_prob < 1.0
            halting = mx.nd.logical_and(halting_prob + p * running > self._threshold, running)
            running = mx.nd.logical_and(halting_prob + p * running <= self._threshold, running)
            halting_prob = halting_prob + p * running
            remainders = remainders + (1 - halting_prob) * halting
            halting_prob = halting_prob + remainders * halting
            updates = updates + running + halting
            weights = (p * running + remainders * halting).expand_dims(2)
            if enc_y is None:
                y, self_attn = fn(state, self_attn_mask)
                self_attns.append(self_attn)
            else:
                y, self_attn, context_attn = fn(state, enc_y, self_attn_mask, context_attn_mask)
                self_attns.append(self_attn)
                context_attns.append(context_attn)
            prev_state = y * weights + prev_state * (1 - weights)
            t += 1
        if enc_y is None:
            return self._layer_norm(prev_state), self_attns
        else:
            return self._layer_norm(prev_state), self_attns, context_attns


if __name__ == "__main__":
    seq = mx.nd.array([[10, 10, 3, 0, 0], [11, 11, 11, 3, 0]])
    seq_len = mx.nd.array([3, 4])
    encoder = Encoder(16, 8, 6, 512, 8, 2048)
    encoder.initialize(mx.init.Xavier())
    enc_y, enc_self_attns = encoder(seq, seq_len)
    print(enc_y, enc_self_attns)
    decoder = Decoder(16, 8, 6, 512, 8, 2048)
    decoder.initialize(mx.init.Xavier())
    dec_y, dec_self_attns, context_attns = decoder(seq, seq_len, enc_y, None)
    print(dec_y, dec_self_attns, context_attns)
