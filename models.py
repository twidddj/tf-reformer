import tensorflow as tf
import numpy as np

from modules import PositionalEncoder, ReformerBlock, pad_len_lsh


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
        random_item = np.random.randint(max_seed, size=2)
        seed1 = random_item[0]
        seed2 = random_item[1]

        with tf.GradientTape(persistent=manual_grad) as tape:
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
