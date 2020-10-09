import tensorflow as tf
import numpy as np

from modules import PositionalEncoder, MultiheadLSHSelfAttention, FeedForward, pad_len_lsh


class ReformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, max_len, attn_config, ff_chunk_size=None, dropout_rate=0.0):
        super(ReformerBlock, self).__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.max_len = max_len
        self.dropout_rate = dropout_rate
        self.ff_chunk_size = ff_chunk_size
        self.seed = None

        self.attn = MultiheadLSHSelfAttention(attn_config, max_len, dropout_rate=dropout_rate)
        self.ff = FeedForward(d_ff, d_model)

    def chunked_ff(self, y1, training=None):
        result = []
        T = y1.shape[1]
        n_chunk = T // self.ff_chunk_size
        chunked_y1 = tf.split(y1, n_chunk, axis=1)
        for _y1 in chunked_y1:
            result.append(self.ff(_y1, training=training))
        return result

    # reversible
    def call(self, x1, x2, t=None, seed=None, training=None):
        y1 = x1 + self.attn(x2, t, seed=seed, training=training)
        if self.ff_chunk_size is None:
            ff_y1 = self.ff(y1, training=training)
        else:
            chunked_ff_y1 = self.chunked_ff(y1, training=training)
            ff_y1 = tf.concat(chunked_ff_y1, axis=1)

        y2 = x2 + ff_y1
        self.seed = seed
        return y1, y2

    def _compute_gradients(self, y1, y2, dy1, dy2):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(y1)
            tape.watch(y2)

            gy1 = self.ff(y1, training=True)
            x2 = y2 - gy1
            fx2 = self.attn(x2, self.max_len, seed=self.seed, training=True)
            x1 = y1 - fx2

        grads_combined = tape.gradient(gy1, [y1] + self.ff.trainable_variables, dy2)
        dx1 = dy1 + grads_combined[0]
        dg = grads_combined[1:]

        grads_combined = tape.gradient(fx2, [x2] + self.attn.trainable_variables, dx1)
        dx2 = dy2 + grads_combined[0]
        df = grads_combined[1:]

        _grads = df + dg
        _vars = self.attn.trainable_variables + self.ff.trainable_variables
        del tape

        return x1, x2, dx1, dx2, _grads, _vars

    def _compute_gradients_chunked(self, y1, y2, dy1, dy2):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(y1)
            tape.watch(y2)

            T = y1.shape[1]
            n_chunk = T // self.ff_chunk_size

            # Split
            chunked_y1 = tf.split(y1, n_chunk, axis=1)
            chunked_y2 = tf.split(y2, n_chunk, axis=1)
            chunked_dy2 = tf.split(dy2, n_chunk, axis=1)

            chunked_x2, chunked_gy1 = [], []

            for _y1, _y2 in zip(chunked_y1, chunked_y2):
                _gy1 = self.ff(_y1, training=True)
                _x2 = _y2 - _gy1
                chunked_gy1.append(_gy1)
                chunked_x2.append(_x2)

            x2 = tf.concat(chunked_x2, axis=1)
            fx2 = self.attn(x2, self.max_len, seed=self.seed, training=True)
            x1 = y1 - fx2

        chunked_dy1, chunked_dg = [], []
        for i in range(len(chunked_x2)):
            _gy1 = chunked_gy1[i]
            _y1 = chunked_y1[i]
            _dy2 = chunked_dy2[i]

            grad_dy1 = tape.gradient(_gy1, [_y1] + self.ff.trainable_variables, _dy2)
            chunked_dy1.append(grad_dy1[0])
            chunked_dg.append(grad_dy1[1:])

        dx1 = dy1 + tf.concat(chunked_dy1, axis=1)
        dg = []

        for j in range(len(chunked_dg[0])):
            item = 0
            for i in range(len(chunked_dg)):
                item += chunked_dg[i][j]
            dg.append(item)

        grads_combined = tape.gradient(fx2, [x2] + self.attn.trainable_variables, dx1)
        dx2 = dy2 + grads_combined[0]
        df = grads_combined[1:]

        _grads = df + dg
        _vars = self.attn.trainable_variables + self.ff.trainable_variables
        del tape

        return x1, x2, dx1, dx2, _grads, _vars

    def compute_gradients(self, y1, y2, dy1, dy2):
        if self.ff_chunk_size is None:
            return self._compute_gradients(y1, y2, dy1, dy2)
        return self._compute_gradients_chunked(y1, y2, dy1, dy2)


class Reformer(tf.keras.Model):
    def __init__(self, d_model, d_ff, vocab_size, max_len, num_blocks, attn_config,
                 ff_chunk_size=None, dropout_rate=0.0):
        super(Reformer, self).__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.dropout_rate = dropout_rate
        self.num_blocks = num_blocks
        self.attn_config = attn_config

        self.embeddings = tf.keras.layers.Embedding(vocab_size, d_model)
        self.positional_encoder = PositionalEncoder(max_len)

        self.blocks = []
        for i in range(num_blocks):
            reformer = ReformerBlock(d_model, d_ff, max_len, attn_config, ff_chunk_size, dropout_rate=dropout_rate)
            self.blocks.append(reformer)

    def to_out(self, x1, x2):
        memory = (x1 + x2) / 2
        return tf.matmul(memory, tf.transpose(self.embeddings.variables[0]))

    def to_emb(self, xs, training=None):
        enc = self.embeddings(xs)
        enc *= self.d_model ** 0.5  # scale
        enc += self.positional_encoder(enc)
        if training:
            enc = tf.nn.dropout(enc, self.dropout_rate)
        return enc

    def call(self, xs, seed=None, training=None):
        if not training:
            cur_len = xs.shape[1]
            pad_num = pad_len_lsh(self.attn_config.bucket_size, cur_len)
            xs = tf.pad(xs, [[0, 0], [0, pad_num]])
        else:
            cur_len = self.max_len

        emb = self.to_emb(xs, training)

        y1, y2 = emb, emb
        for block in self.blocks:
            y1, y2 = block(y1, y2, cur_len, seed=seed, training=training)

        return emb, y1, y2

    def ar_gen(self, xs):
        cur_len = xs.shape[1]
        _, y1, y2 = self.call(xs, training=False)
        logits = self.to_out(y1, y2)
        y_pred = tf.argmax(logits[:, cur_len - 1], -1)
        return y_pred

    def compute_gradients(self, tape, emb, y1, y2, loss):
        grads_list = []
        vars_list = []
        emb_var = self.embeddings.trainable_variables[0]

        _grads = tape.gradient(loss, [y1, y2, emb_var])
        dy1, dy2 = _grads[0], _grads[1]
        _grads = _grads[2:]

        grads_list.extend(_grads)
        vars_list.append(emb_var)

        y1, y2, dy1, dy2, _grads, _vars = self._compute_gradients(y1, y2, dy1, dy2)
        grads_list.extend(_grads)
        vars_list.extend(_vars)

        d_emb = tf.convert_to_tensor(tape.gradient(emb, emb_var, dy1))
        d_emb += tf.convert_to_tensor(tape.gradient(emb, emb_var, dy2))
        grads_list[0] += d_emb

        del tape

        grad_and_vars = zip(grads_list, vars_list)
        return grad_and_vars

    def _compute_gradients(self, y1, y2, dy1, dy2):
        grads_all = []
        vars_all = []

        for i in reversed(range(len(self.blocks))):
            block = self.blocks[i]
            y1, y2, dy1, dy2, _grads, _vars = block.compute_gradients(y1, y2, dy1, dy2)
            grads_all.extend(_grads)
            vars_all.extend(_vars)

        return y1, y2, dy1, dy2, grads_all, vars_all

    @tf.function
    def train_step(self, xs, labels, loss_func, optimizer, manual_grad=True, max_seed=2**32):
        if manual_grad:
            random_item = np.random.randint(max_seed, size=2)
            seed1 = random_item[0]
            seed2 = random_item[1]
        else:
            seed2 = None

        with tf.GradientTape(persistent=manual_grad) as tape:
            if manual_grad:
                tf.random.set_seed(seed1)
            emb, y1, y2 = self.call(xs, seed=seed2, training=True)

            if manual_grad:
                y1, y2 = tf.stop_gradient(y1), tf.stop_gradient(y2)
                tape.watch(y1)
                tape.watch(y2)
                
            logits = self.to_out(y1, y2)
            loss, y_pred = loss_func(logits, labels)

        if manual_grad:
            tf.random.set_seed(seed1)
            grad_and_vars = self.compute_gradients(tape, emb, y1, y2, loss)
        else:
            grads = tape.gradient(loss, self.trainable_variables)
            grad_and_vars = zip(grads, self.trainable_variables)

        del tape

        optimizer.apply_gradients(grad_and_vars)

        return loss, y_pred
