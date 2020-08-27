"""
modules for reformer
codes are borrowed from
https://github.com/lucidrains/reformer-pytorch
https://github.com/cerebroai/reformers
https://github.com/renmengye/revnet-public
"""

import tensorflow as tf
from transformer_modules import mask, ln


def sort_key_val(t1, t2, N, dim=-1):
    ids = tf.argsort(t1, axis=dim)
    values = tf.gather(t1, ids, batch_dims=1)
    # t2 = tf.broadcast_to(t2, t1.shape)
    t2 = tf.tile(t2, [N, 1])
    return values, tf.gather(t2, ids, batch_dims=1)


def batched_index_select(values, indices):
    return tf.squeeze(tf.gather(values, indices, batch_dims=1))


def make_unit_length(x, epsilon=1e-6):
    norm = tf.norm(x, ord=2, axis=-1, keepdims=True)
    return tf.math.truediv(x, norm + epsilon)


def mask_out(x, mask, mask_val=float('-inf')):
    present = tf.cast(1 - tf.cast(mask, tf.int32), tf.bool)
    mask = tf.cast(mask, tf.float32)
    x = tf.where(present, x, mask * mask_val)
    return x


def hash_vec(x, x_len, num_hashes, bucket_size, dropout_rate=0, training=True):
    N, T, dim = x.shape

    n_buckets = x_len // bucket_size
    rot_size = n_buckets

    # Hashing
    rotations_shape = (1, dim, num_hashes, rot_size // 2)
    random_rotations = tf.random.normal(rotations_shape)
    x = tf.layers.dropout(x, rate=dropout_rate, training=training)

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


def lsh_attention(qk, v, T, num_hashes=2, bucket_size=4, is_full=False, input_mask=None,
                  dropout_rate=0, training=True, causality=False):
    N, _, dim = qk.shape

    if is_full:
        # full attn
        buckets = tf.zeros((N, T), tf.int64)
        n_buckets = 1
        num_hashes = 1
    else:
        buckets = hash_vec(qk, T, num_hashes, bucket_size, dropout_rate=dropout_rate, training=training)
        n_buckets = T // bucket_size

    """
    For preserving temporal order when it sorted.
    let a hash bucket := [0, 1, 1, 0, 0, 1], T=6
    multiply [0, 1, 1, 0, 0, 1] by 6 -> [0, 6, 6, 0, 0, 6]
    [0, 6, 6, 0, 0, 6] + [0, 1, 2, 3, 4, 5] = [0, 7, 8, 3, 4, 11]
    the bucket after sorted  [0, 3, 4, 7, 8, 11]
    """
    ticker = tf.expand_dims(tf.range(num_hashes * T), axis=0)
    buckets_and_t = T * buckets + tf.cast((ticker % T), tf.int64)
    buckets_and_t = tf.stop_gradient(buckets_and_t)
    sbuckets_and_t, sticker = sort_key_val(buckets_and_t, ticker, N, dim=-1)

    """
    It needs to undo sort after attention operation for each hash bucket.
    # 해시버킷 별 attention 후 원래 순서로 복원
    """
    _, undo_sort = sort_key_val(sticker, ticker, N, dim=-1)

    sticker = tf.stop_gradient(sticker)
    undo_sort = tf.stop_gradient(undo_sort)

    """
    Sorted QK
    Sorted V
    # 정렬된 hash 인덱스를 이용해서 데이터 개더링
    """
    st = sticker % T
    sqk = qk if is_full else batched_index_select(qk, st)
    sv = v if is_full else batched_index_select(v, st)

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
    if not is_full:
        def look_one_back(x):
            x_extra = tf.concat([x[:, -1:, ...], x[:, :-1, ...]], axis=1)
            return tf.concat([x, x_extra], axis=2)
        bk = look_one_back(bk)
        bv = look_one_back(bv)
        bkv_t = look_one_back(bkv_t)

    # Dot-product attention.
    dots = tf.einsum('bhie,bhje->bhij', bq, bk) * (tf.cast(bq.shape[-1], tf.float32) ** -0.5)

    """
    This is for masking different hash vectors in a chunk.
    # 다른 해시 값일 경우 마스킹 처리 하기 위한 코드
    # 어차피 청크 내 모든 벡터들에 대해 계산을 해야되기 때문에 꼭 필요하지는 않은 것 같음
    """
    if not is_full:
        q_sbuckets = tf.gather(buckets, sticker, batch_dims=1)
        q_sbuckets = tf.reshape(q_sbuckets, (N, chunk_size, -1))
        kv_sbuckets = look_one_back(q_sbuckets)
        mask = 1 - tf.cast(tf.equal(q_sbuckets[:, :, :, None], kv_sbuckets[:, :, None, :]), tf.int32)
        dots = mask_out(dots, mask)

    if input_mask is not None:
        mq = tf.gather(input_mask, st, batch_dims=1)
        mq = tf.reshape(mq, (N, chunk_size, -1))
        mq = tf.cast(mq, tf.int32)
        if not is_full:
            mkv = look_one_back(mq)
            mask = (1 - mq[:, :, :, None] * mkv[:, :, None, :])
        else:
            mask = (1 - mq[:, :, :, None] * mq[:, :, None, :])
        dots = mask_out(dots, mask)

    # Causal masking
    if causality:
        mask = tf.greater(bkv_t[:, :, None, :], bq_t[:, :, :, None])
        dots = mask_out(dots, mask)

    # Mask out attention to self except when no other targets are available.
    mask = tf.equal(bq_t[:, :, :, None], bkv_t[:, :, None, :])
    dots = mask_out(dots, mask, mask_val=-1e-5)
    del mask

    # normalize dots on each bucket
    dots_logsumexp = tf.math.reduce_logsumexp(dots, axis=-1, keepdims=True)
    dots = tf.exp(dots - dots_logsumexp)
    dots = tf.layers.dropout(dots, rate=dropout_rate, training=training)

    # weighted sum
    bo = tf.einsum('buij,buje->buie', dots, bv)
    so = tf.reshape(bo, (N, -1, bo.shape[-1]))
    slogits = tf.reshape(dots_logsumexp, (N, -1,))

    # undo sort
    o = so if is_full else batched_index_select(so, undo_sort)
    o = tf.reshape(o, (N, num_hashes, -1, qk.shape[-1]))
    logits = slogits if is_full else batched_index_select(slogits, undo_sort)
    logits = tf.reshape(logits, (N, num_hashes, -1, 1))

    # normalize outputs on each hash
    probs = tf.exp(logits - tf.math.reduce_logsumexp(logits, axis=1, keepdims=True))
    out = tf.reduce_sum(o * probs, 1)
    return out


def multihead_lsh_attention(queries, keys, values, seq_len=None,
                            is_full=False,
                            max_seq_len=None,
                            num_hashses=2,
                            bucket_size=4,
                            num_heads=8,
                            dropout_rate=0,
                            training=True,
                            causality=False,
                            scope="multihead_lsh_attention"):

    N, T, d_model = queries.shape
    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
        # Linear projections, Q=K
        with tf.compat.v1.variable_scope('qk_prj', reuse=tf.compat.v1.AUTO_REUSE):
            Q = tf.layers.dense(queries, d_model, use_bias=True)  # (N, T_q, d_model)
        V = tf.layers.dense(values, d_model, use_bias=True)

        # Split and concat
        Q_ = tf.split(Q, num_heads, axis=2)
        V_ = tf.split(V, num_heads, axis=2)

        input_masks = None
        if seq_len is not None:
            input_mask = tf.sequence_mask(seq_len, max_seq_len)
            input_mask = tf.expand_dims(input_mask, 0)
            input_masks = tf.tile(input_mask, [N, 1])

            # assert seq_len % bucket_size == 0
            tT = bucket_size
            seq_len += (tT - (seq_len % tT)) % tT

        outputs = []
        for qk, v in zip(Q_, V_):
            outputs.append(lsh_attention(qk, v, seq_len,
                                         num_hashes=num_hashses, bucket_size=bucket_size, input_mask=input_masks,
                                         dropout_rate=dropout_rate, training=training, causality=causality, is_full=is_full))

        outputs = tf.concat(outputs, -1)

        # Normalize
        outputs = ln(outputs)
    return outputs


def ff(x, num_units, scope="positionwise_feedforward"):
    d_ff, d_model = num_units
    assert d_ff % d_model == 0
    n_chunk = d_ff // d_model
    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
        W1 = tf.compat.v1.get_variable('W1', [d_model, d_ff])
        B1 = tf.compat.v1.get_variable('B1', [d_ff])
        W2 = tf.compat.v1.get_variable('W2', [d_ff, d_model])
        B2 = tf.compat.v1.get_variable('B2', [d_model])

        # naive chunking
        outputs = tf.zeros_like(x)
        for i in range(n_chunk):
            w1 = tf.slice(W1, [0, i * d_model], [-1, d_model])
            b1 = tf.slice(B1, [i * d_model], [d_model])
            h0 = tf.nn.relu(tf.matmul(x, w1) + b1)
            w2 = tf.slice(W2, [i * d_model, 0], [d_model, -1])
            outputs += tf.matmul(h0, w2)
        outputs += B2

        # Normalize
        outputs = ln(outputs)

    return outputs


class ReversibleSequence:
    def __init__(self, blocks):
        self.blocks = blocks
        self.name_tmpl = "num_blocks_{}"

    def __call__(self, x1, x2):
        for i, block in enumerate(self.blocks):
            name = self.name_tmpl.format(i)
            with tf.compat.v1.variable_scope(name):
                x1, x2 = block(x1, x2)
                block.f.t_vars = tf.compat.v1.trainable_variables(name + "/multihead_lsh_attention")
                block.g.t_vars = tf.compat.v1.trainable_variables(name + "/positionwise_feedforward")
        return x1, x2

    def compute_gradients(self, y1, y2, dy1, dy2):
        grads_all = []
        vars_all = []

        for i in reversed(range(len(self.blocks))):
            block = self.blocks[i]
            name = self.name_tmpl.format(i)
            with tf.compat.v1.variable_scope(name, reuse=True):
                y1, y2, dy1, dy2, _grads, _vars = block.compute_gradients(y1, y2, dy1, dy2)
                grads_all.extend(_grads)
                vars_all.extend(_vars)

        return y1, y2, dy1, dy2, grads_all, vars_all


class ReversibleBlock:
    def __init__(self, f, g):
        self.f = f
        self.g = g
        self.t_vars = None

    def __call__(self, x1, x2):
        y1 = x1 + self.f(x2)
        y2 = x2 + self.g(y1)
        return y1, y2

    def compute_gradients(self, y1, y2, dy1, dy2):
        gy1 = self.g(y1)
        x2 = y2 - gy1
        fx2 = self.f(x2)
        x1 = y1 - fx2

        grads_combined = tf.gradients(gy1, [y1] + self.g.t_vars, dy2, gate_gradients=True)
        dx1 = dy1 + grads_combined[0]
        dg = grads_combined[1:]

        grads_combined = tf.gradients(fx2, [x2] + self.f.t_vars, dx1, gate_gradients=True)
        dx2 = dy2 + grads_combined[0]
        df = grads_combined[1:]

        _grads = df + dg
        _vars = self.f.t_vars + self.g.t_vars

        with tf.control_dependencies(_grads):
            dy1, dy2 = tf.identity(dx1), tf.identity(dx2)

        return x1, x2, dy1, dy2, _grads, _vars