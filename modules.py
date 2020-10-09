"""
modules for reformer
some codes are borrowed from
https://github.com/lucidrains/reformer-pytorch
https://github.com/cerebroai/reformers
https://github.com/renmengye/revnet-public
"""

import tensorflow as tf
import numpy as np


def sort_key_val(t1, t2, dim=-1):
    ids = tf.argsort(t1, axis=dim)
    values = tf.gather(t1, ids, batch_dims=1)
    return values, tf.gather(t2, ids, batch_dims=1)


def batched_index_select(values, indices):
    return tf.squeeze(tf.gather(values, indices, batch_dims=1))


def make_unit_length(x, epsilon=1e-6):
    norm = tf.norm(x, ord=2, axis=-1, keepdims=True)
    return tf.math.truediv(x, norm + epsilon)


def mask_out(x, mask, mask_val=float('-inf')):
    present = tf.math.logical_not(mask)
    mask = tf.cast(mask, tf.float32)
    x = tf.where(present, x, mask * mask_val)
    return x


def hash_vec(x, x_len, num_hashes, bucket_size, seed=None, dropout_rate=0, training=True):
    N, T, dim = x.shape

    n_buckets = x_len // bucket_size
    rot_size = n_buckets

    # Hashing
    rotations_shape = (1, dim, num_hashes, rot_size // 2)
    random_rotations = tf.random.normal(rotations_shape, seed=seed)
    random_rotations = tf.tile(random_rotations, [N, 1, 1, 1])
    if training:
        x = tf.nn.dropout(x, dropout_rate)

    rotated_vecs = tf.einsum('btf,bfhi->bhti', x, random_rotations)
    rotated_vecs = tf.concat([rotated_vecs, -rotated_vecs], axis=-1)  # N x num_hashes x T x rot_size
    tmp = tf.math.argmax(rotated_vecs, axis=-1)

    """
    add offset so that each hash can be distinguished in multiround LSH
    # multiround LSH를 수행할 때, 각 hash bucket을 구별하여 정렬할 수 있도록 offset을 더해줌
    """
    offsets = tf.range(num_hashes, dtype=tf.int64)
    offsets = tf.reshape(offsets * n_buckets, (1, -1, 1))
    offsets = tf.cast(offsets, tf.int64)
    buckets = tf.reshape(tmp + offsets, [N, -1])  # N x (num_hashes*T)

    return buckets


def lsh_attention(qk, v, T, seed=None, num_hashes=2, bucket_size=4, use_full=False, input_mask=None,
                  dropout_rate=0, training=True, causality=False, causal_start=None):
    N, _, dim = qk.shape

    if use_full:
        # full attn
        buckets = tf.zeros((N, T), tf.int64)
        n_buckets = 1
        num_hashes = 1
    else:
        buckets = hash_vec(qk, T, num_hashes, bucket_size, seed=seed, dropout_rate=dropout_rate, training=training)
        n_buckets = T // bucket_size

    """
    For preserving temporal order when it sorted.
    let a hash bucket := [0, 1, 1, 0, 0, 1], T=6
    multiply [0, 1, 1, 0, 0, 1] by 6 -> [0, 6, 6, 0, 0, 6]
    [0, 6, 6, 0, 0, 6] + [0, 1, 2, 3, 4, 5] = [0, 7, 8, 3, 4, 11]
    the bucket after sorted  [0, 3, 4, 7, 8, 11]
    """
    ticker = tf.expand_dims(tf.range(num_hashes * T), axis=0)
    ticker = tf.tile(ticker, [N, 1])

    if use_full:
        buckets_and_t, sbuckets_and_t, sticker = ticker, ticker, ticker
    else:
        buckets_and_t = T * buckets + tf.cast((ticker % T), tf.int64)
        buckets_and_t = tf.stop_gradient(buckets_and_t)
        sbuckets_and_t, sticker = sort_key_val(buckets_and_t, ticker, dim=-1)

    """
    It needs to undo sort after attention operation for each hash bucket.
    # 해시버킷 별 attention 후 원래 순서로 복원
    """
    _, undo_sort = sort_key_val(sticker, ticker, dim=-1)

    """
    No need to store the memory of gradients for these variables
    # 이 변수들에 대해서는 그라디언트 메모리를 가지고 있을 필요가 없음
    """
    sticker = tf.stop_gradient(sticker)
    undo_sort = tf.stop_gradient(undo_sort)

    """
    Sorted QK
    Sorted V
    # 정렬된 hash 인덱스를 이용해서 데이터 개더링
    """
    st = sticker % T
    sqk = qk if use_full else batched_index_select(qk, st)
    sv = v if use_full else batched_index_select(v, st)

    """  
    # 버킷 별로 데이터를 reshape
    # T=20 이고 버킷크기가 4라면 N x 5 x 4 x dim 으로 변환 (4짜리 버킷 5개)
    """
    chunk_size = num_hashes * n_buckets
    bq_t = bkv_t = tf.reshape(st, (N, chunk_size, -1))
    bqk = tf.reshape(sqk, (N, chunk_size, -1, dim))
    bv = tf.reshape(sv, (N, chunk_size, -1, dim))

    # Hashing operates on unit-length vectors. Unnormalized query vectors are
    # fine because they effectively provide a learnable temperature for the
    # attention softmax, but normalizing keys is needed so that similarity for
    # the purposes of attention correctly corresponds to hash locality.
    bq = bqk
    bk = make_unit_length(bqk)

    # TODO: Parameterized the number of previous chunks.
    """
    Here, only 1 previous chunk can be considered in attention operation.
    Although the chunk at the starting boundary gets a hashed chunk that is different from itself,
    The chunks will be masked out.
    # 단 한 개의 이전 chunk를 attend할 수 있게
    # 시작 경계의 벡터는 다르게 해시된 chunk를 가져 오지만 어차피 마스킹 되므로 노 상관
    """
    if not use_full:
        def look_one_back(x):
            x_extra = tf.concat([x[:, -1:, ...], x[:, :-1, ...]], axis=1)
            return tf.concat([x, x_extra], axis=2)

        bk = look_one_back(bk)
        bv = look_one_back(bv)
        bkv_t = look_one_back(bkv_t)

    # Dot-product attention.
    # batch x (bucket_size * num_hashes) x bucket_size x (bucket_size * 2(look_one_back))
    dots = tf.einsum('bhie,bhje->bhij', bq, bk) * (tf.cast(bq.shape[-1], tf.float32) ** -0.5)

    """
    This is for masking different hash vectors in a chunk.
    # 다른 해시 값일 경우 마스킹 처리 하기 위한 코드
    # 어차피 청크 내 모든 벡터들에 대해 계산을 해야되기 때문에 꼭 필요하지는 않은 것 같음
    """
    if not use_full:
        q_sbuckets = tf.gather(buckets, sticker, batch_dims=1)
        q_sbuckets = tf.reshape(q_sbuckets, (N, chunk_size, -1))
        kv_sbuckets = look_one_back(q_sbuckets)
        mask = tf.logical_not(tf.equal(q_sbuckets[:, :, :, None], kv_sbuckets[:, :, None, :]))
        dots = mask_out(dots, mask)

    if input_mask is not None:
        mq = tf.gather(input_mask, st, batch_dims=1)
        mq = tf.reshape(mq, (N, chunk_size, -1))
        mq = tf.cast(mq, tf.int32)
        if not use_full:
            mkv = look_one_back(mq)
            mask = (1 - mq[:, :, :, None] * mkv[:, :, None, :])
        else:
            mask = (1 - mq[:, :, :, None] * mq[:, :, None, :])
        mask = tf.cast(mask, tf.bool)
        dots = mask_out(dots, mask)

    # Causal masking
    if causality:
        if causal_start is None:
            mask = tf.greater(bkv_t[:, :, None, :], bq_t[:, :, :, None])
        else:
            _bkv_t = tf.where(bkv_t >= causal_start, bkv_t, 0)
            _bq_t = tf.where(bq_t >= causal_start, bq_t, 0)
            mask = tf.greater(_bkv_t[:, :, None, :], _bq_t[:, :, :, None])  # bkv_t > bq_t

        dots = mask_out(dots, mask)

    # Mask out attention to self except when no other targets are available.
    mask = tf.equal(bq_t[:, :, :, None], bkv_t[:, :, None, :])
    dots = mask_out(dots, mask, mask_val=-1e-5)
    del mask

    # normalize dots on each bucket
    dots_logsumexp = tf.math.reduce_logsumexp(dots, axis=-1, keepdims=True)
    dots = tf.exp(dots - dots_logsumexp)
    if training:
        dots = tf.nn.dropout(dots, dropout_rate)

    # weighted sum
    bo = tf.einsum('buij, buje->buie', dots, bv)
    so = tf.reshape(bo, (N, -1, bo.shape[-1]))
    slogits = tf.reshape(dots_logsumexp, (N, -1,))

    # undo sort
    o = so if use_full else batched_index_select(so, undo_sort)
    o = tf.reshape(o, (N, num_hashes, -1, qk.shape[-1]))
    logits = slogits if use_full else batched_index_select(slogits, undo_sort)
    logits = tf.reshape(logits, (N, num_hashes, -1, 1))

    # normalize outputs on each hash
    probs = tf.exp(logits - tf.math.reduce_logsumexp(logits, axis=1, keepdims=True))
    out = tf.reduce_sum(o * probs, 1)
    return out


def pad_len_lsh(bs, seq_len):
    return (bs - (seq_len % bs)) % bs


class Config:
    def __init__(self, _dict):
        self.__dict__ = _dict


class PositionalEncoder(tf.keras.layers.Layer):
    def __init__(self, maxlen, masking=False, mask_val=None):
        super(PositionalEncoder, self).__init__()
        self.maxlen = maxlen
        self.masking = masking
        self.mask_val = mask_val

    def build(self, input_shape):
        _, _, D = input_shape

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i - i % 2) / D) for i in range(D)]
            for pos in range(self.maxlen)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        self.params = tf.convert_to_tensor(position_enc, tf.float32)  # (maxlen, E)

    def call(self, inputs):
        N, T, _ = inputs.shape

        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])  # (N, T)
        outputs = tf.nn.embedding_lookup(self.params, position_ind)

        # masks
        if self.masking:
            assert self.mask_val is not None
            outputs = tf.where(tf.equal(inputs, self.mask_val), 0.0, outputs)

        return outputs


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_ff, d_model):
        super(FeedForward, self).__init__()
        assert (d_ff % d_model) == 0
        self.d_ff = d_ff
        self.d_model = d_model
        self.n_chunk = d_ff // d_model

        self.ln = tf.keras.layers.LayerNormalization()

    def build(self, input_shape):
        dim = input_shape[-1]
        self.W1 = self.add_weight(name='W1', shape=[dim, self.d_ff], trainable=True)
        self.B1 = self.add_weight(name='B1', shape=[self.d_ff], trainable=True)
        self.W2 = self.add_weight(name='W2', shape=[self.d_ff, self.d_model], trainable=True)
        self.B2 = self.add_weight(name='B2', shape=[self.d_model], trainable=True)

    def call(self, inputs):
        outputs = tf.zeros_like(inputs)
        for i in range(self.n_chunk):
            w1 = tf.slice(self.W1, [0, i * self.d_model], [-1, self.d_model])
            b1 = tf.slice(self.B1, [i * self.d_model], [self.d_model])
            h0 = tf.nn.relu(tf.matmul(inputs, w1) + b1)
            w2 = tf.slice(self.W2, [i * self.d_model, 0], [self.d_model, -1])
            outputs += tf.matmul(h0, w2)
        outputs += self.B2

        outputs = self.ln(outputs)
        return outputs


class MultiheadLSHSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config, max_len, dropout_rate=0.0):

        super(MultiheadLSHSelfAttention, self).__init__()

        self.config = config
        self.max_len = max_len
        self.dropout_rate = dropout_rate
        self.to_Q = tf.keras.layers.Dense(config.dim)
        self.to_V = tf.keras.layers.Dense(config.dim)
        self.ln = tf.keras.layers.LayerNormalization()

    def call(self, inputs, seq_len=None, seed=None, training=None):
        N, T, _ = inputs.shape

        Q = self.to_Q(inputs)
        V = self.to_V(inputs)

        # Split
        Q_ = tf.split(Q, self.config.num_heads, axis=2)
        V_ = tf.split(V, self.config.num_heads, axis=2)

        input_masks = None

        # AR생성에서 실제 seq_len 이후 데이터는 마스크 되어야 함
        if not training:
            assert seq_len is not None
            input_mask = tf.sequence_mask(seq_len, self.max_len)
            input_mask = tf.expand_dims(input_mask, 0)
            input_masks = tf.tile(input_mask, [N, 1])

            seq_len += pad_len_lsh(self.config.bucket_size, seq_len)
        else:
            # training 중 seq_len = 최대 시퀀스 길이
            seq_len = T

        outputs = []
        for qk, v in zip(Q_, V_):
            outputs.append(lsh_attention(qk, v, seq_len,
                                         seed=seed,
                                         num_hashes=self.config.num_hashes,
                                         bucket_size=self.config.bucket_size,
                                         input_mask=input_masks,
                                         dropout_rate=self.dropout_rate,
                                         training=training,
                                         causality=self.config.causality,
                                         causal_start=self.config.causal_start,
                                         use_full=self.config.use_full))

        outputs = tf.concat(outputs, -1)
        outputs = self.ln(outputs)

        return outputs